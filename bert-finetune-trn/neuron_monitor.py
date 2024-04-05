from metaflow.cards import Markdown, Table, VegaChart
from functools import wraps
import threading
from datetime import datetime
from metaflow import current
from metaflow.cards import Table, Markdown, VegaChart, Image, Artifact
import time
import shutil

import re
import os
import uuid
import json
import sys
import select
from tempfile import TemporaryFile, TemporaryDirectory
from subprocess import check_output, Popen, PIPE
from datetime import datetime, timedelta
from functools import wraps
from collections import namedtuple

# Card plot styles
MEM_COLOR = "#0c64d6"
NEURON_COLOR = "#ff69b4"

TS_FORMAT = "%Y/%m/%d %H:%M:%S"

MONITOR_FIELDS = [
    "timestamp",
    "neuron_utilization",
    "memory_used",
    "memory_total",
]

CONFIG_FILENAME = "monitor.conf"
MONITOR = """neuron-monitor --config-file {}""".format(CONFIG_FILENAME)
CONFIG_TEMPLATE = {
    "period": None,
    "neuron_runtimes": [
        {
            "tag_filter": ".*",
            "metrics": [
                {"type": "neuroncore_counters"},
                {"type": "memory_used"},
                {"type": "neuron_runtime_vcpu_usage"},
                {"type": "execution_stats"},
            ],
        }
    ],
    "system_metrics": [
        {"type": "vcpu_usage"},
        {"type": "memory_info"},
        {"type": "neuron_hw_counters"},
    ],
}

ProcessUUID = namedtuple("ProcessUUID", ["uuid", "start_time", "end_time"])


def _get_uuid(time_duration=600):
    frmt_str = "%Y-%m-%d-%H-%M-%S"
    # Create a datetime range between the timerange values using current date as start date and time_duration as end date
    start_date = datetime.now()
    end_date = start_date + timedelta(seconds=time_duration)
    datetime_range = start_date = (
        datetime.now().strftime(frmt_str) + "_" + end_date.strftime(frmt_str)
    )
    uuid_str = uuid.uuid4().hex.replace("-", "") + "_" + datetime_range
    return ProcessUUID(uuid_str, start_date, end_date)


class AsyncProcessManager:
    """
    This class is responsible for managing the neuron monitor subprocess
    """

    processes = {
        # "procid": {
        #     "proc": subprocess.Popen,
        #     "started": time.time()
        # }
    }

    @classmethod
    def _register_process(cls, procid, proc):
        cls.processes[procid] = {
            "proc": proc,
            "started": time.time(),
        }

    @classmethod
    def get(cls, procid):
        proc_dict = cls.processes.get(procid, None)
        if proc_dict is not None:
            return proc_dict["proc"], proc_dict["started"]
        return None, None

    @classmethod
    def spawn(cls, procid, cmd, file):
        proc = Popen(cmd, stdout=file)
        cls._register_process(procid, proc)

    @classmethod
    def remove(cls, procid, delete_item=True):
        if procid in cls.processes:
            if cls.processes[procid]["proc"].stdout is not None:
                cls.processes[procid]["proc"].stdout.close()
            cls.processes[procid]["proc"].terminate()
            cls.processes[procid]["proc"].wait()
            if delete_item:
                del cls.processes[procid]

    @classmethod
    def cleanup(cls):
        for procid in cls.processes:
            cls.remove(procid, delete_item=False)
        cls.processes.clear()

    @classmethod
    def is_running(cls, procid):
        if procid not in cls.processes:
            return False
        return cls.processes[procid]["proc"].poll() is None


def _parse_timestamp(timestamp):
    try:
        ts = timestamp.split(".")[0]
        return datetime.strptime(ts, TS_FORMAT)
    except ValueError:
        return None


class NeuronMonitor:
    """
    The `NeuronMonitor` class is designed to monitor Neuron core usage.

    When an instance of `NeuronMonitor` is created, it initializes with a specified `interval` and `duration`.
    The `duration` is the timeperiod it will run the neuron-monitor command for and the `interval` is the timeperiod between each reading.
    The class exposes a `_monitor_update_thread` method which runs as a background thread that continuously updates the Neuron usage readings.
    It will keep running until the `_finished` flag is set to `True`.

    The class will statefully manage the the spawned neuron-monitor processes.
    It will start a new neuron-monitor process after the current one has ran for the specified `duration`.
    At a time this class will only maintain readings for the `_current_process` and will have all the aggregated
    readings for the past processes stored in the `_past_readings` dictionary.
    When a process finishes completion, the readings are appended to the `_past_readings` dictionary and a new process is started.

    If the caller of this class wishes to read the Neuron usage, they can call the `read` method which will return the readings in a dictionary format.
    The `read` method will aggregate the readings from the `_current_readings` and `_past_readings`.
    """

    _started_processes = []

    _current_process: ProcessUUID = None

    _current_readings = {}

    _past_readings = {}

    def __init__(self, interval=1, duration=300) -> None:

        with open(CONFIG_FILENAME, "w") as f:
            CONFIG_TEMPLATE["period"] = f"{interval}s"
            json.dump(CONFIG_TEMPLATE, f)

        self._tempdir = TemporaryDirectory(prefix="neuron_card_monitor", dir="./")
        self._interval = interval
        self._duration = duration
        self._finished = False

    @property
    def _current_file(self):
        if self._current_process is None:
            return None
        return os.path.join(self._tempdir.name, self._current_process.uuid + ".log")

    def get_file_name(self, uuid):
        return os.path.join(self._tempdir.name, uuid + ".log")

    def create_new_monitor(self):
        uuid = _get_uuid(self._duration)
        # file = open(self.get_file_name(uuid.uuid), "w")
        file = PIPE
        cmd = MONITOR
        AsyncProcessManager.spawn(uuid.uuid, ["bash", "-c", cmd], file)
        self._started_processes.append(uuid)
        self._poller = select.poll()
        proc = AsyncProcessManager.get(uuid.uuid)[0]
        self._poller.register(proc.stdout, select.POLLIN)
        self._current_process = uuid
        return uuid

    def clear_current_monitor(self):
        if self._current_process is None:
            return
        AsyncProcessManager.remove(self._current_process.uuid)
        self._current_process = None

    def current_process_has_ended(self):
        if self._current_process is None:
            return True
        return datetime.now() > self._current_process.end_time

    def current_process_is_running(self):
        if self._current_process is None:
            return False
        return AsyncProcessManager.is_running(self._current_process.uuid)

    def _read_monitor(self):
        """
        Reads the monitor file and returns the readings in a dictionary format
        """
        devdata = {
            # <device_id>: {
            #     "neuron_utilization": [],
            #     "memory_used": [],
            #     "memory_total": [],
            #     "timestamp": []
            # }
        }

        if self._poller.poll(1):
            proc = AsyncProcessManager.get(self._current_process.uuid)[0]
            try:
                if proc.stdout is not None:
                    line = proc.stdout.readline()
                    jtmp = json.loads(line.decode())
                    time = datetime.now().strftime(TS_FORMAT)
                    vals = {}
                    for rt_data in jtmp["neuron_runtime_data"]:
                        # core utilization
                        d = rt_data["report"]["neuroncore_counters"]["neuroncores_in_use"]
                        for k in d.keys():
                            vals[k] = vals.get(k, 0) + d[k]["neuroncore_utilization"]
                            if k not in devdata:
                                devdata[k] = {
                                    "neuron_utilization": [],
                                    "memory_used": [],
                                    "memory_total": [],
                                    "timestamp": [],
                                }
                            devdata[k]["neuron_utilization"].append(vals[k])
                        # memory utilization
                        d = rt_data["report"]["memory_used"]["neuron_runtime_used_bytes"][
                            "usage_breakdown"
                        ]["neuroncore_memory_usage"]
                        for k in d.keys():
                            devdata[k]["memory_used"].append(sum(d[k].values()))
                            devdata[k]["memory_total"].append(
                                32 / 2 * 1024 * 1024 * 1024
                            )  # NOTE: assumes 32GB memory per device.
                        # metadata
                        for k in d.keys():
                            devdata[k]["timestamp"].append(time)
            except AttributeError as e:
                pass # Ignore the error, it means there is no process running to read from
        return devdata

    def _update_readings(self):
        """
        Core update function that checks if the current process has ended and if so, it will create a new monitor
        otherwise sets the current readings to the readings from the monitor file
        """
        if self.current_process_has_ended() or not self.current_process_is_running():
            self._update_past_readings()
            self.clear_current_monitor()
            uuid = self.create_new_monitor()
            # Sleep for 1 second to allow the new process to start and we can make a reading
            time.sleep(1)

        readings = self._read_monitor()
        if readings is None:
            return
        self._current_readings = readings

    @staticmethod
    def _make_full_reading(current, past):
        device_id = "0"
        if current is None:
            return past
        for device_id in current:
            if device_id not in past:
                past[device_id] = {}
            for field in MONITOR_FIELDS:
                if field not in past[device_id]:
                    past[device_id][field] = []
                past[device_id][field].extend(current[device_id][field])
        return past

    def read(self):
        return self._make_full_reading(
            self._current_readings, json.loads(json.dumps(self._past_readings))
        )

    def read_hardware_info(self):
        """
        Will read the hardware info from the monitor and return it as a dictionary.
        It should run before meaningful data is available from the monitor readings.
        """
        if self._poller.poll(1):
            proc = AsyncProcessManager.get(self._current_process.uuid)[0]
            line = proc.stdout.readline()
            data = json.loads(line)
            if "neuron_hardware_info" in data:
                return data["neuron_hardware_info"]

    def _update_past_readings(self):
        if self._current_readings is None:
            return
        self._past_readings = self._make_full_reading(
            self._current_readings, json.loads(json.dumps(self._past_readings))
        )
        self._current_readings = None

    def cleanup(self):
        self._finished = True
        AsyncProcessManager.cleanup()
        self._tempdir.cleanup()

    def _monitor_update_thread(self):
        while not self._finished:
            self._update_readings()
            time.sleep(self._interval)


def _get_ts_range(_range):
    if _range == "":
        return "*No readings available*"
    return "*Time range of charts: %s*" % _range


def _update_utilization(results, md_dict):
    for device, data in results["profile"].items():
        if device not in md_dict:
            print(
                "Device %s not found in the Neuron monitor card layout. Skipping..."
                % device,
                file=sys.stderr,
            )
            continue
        if ("neuron_utilization" in data) and ("memory_used" in data):
            md_dict[device]["neuron"].update(
                "%2.1f%%" % max(map(float, data["neuron_utilization"]))
            )
            md_dict[device]["memory"].update(
                "%dMB" % max(map(lambda x: float(x) / (1024 * 1024), data["memory_used"]))
            )


def _update_charts(results, md_dict):
    for device, data in results["profile"].items():
        try:
            if device not in md_dict:
                continue
            if ("neuron_utilization" not in data) or ("memory_used" not in data):
                continue
            neuron_plot, mem_plot, ts_range = profile_plots(
                device,
                data["timestamp"],
                data["neuron_utilization"],
                data["memory_used"],
                data["memory_total"],
            )
            md_dict[device]["neuron"].update(neuron_plot)
            md_dict[device]["memory"].update(mem_plot)
            md_dict[device]["reading_duration"].update(_get_ts_range(ts_range))
        except ValueError as e:
            # This is thrown when the date is unparsable. We can just safely ignore this.
            print("ValueError: Could not parse date \n%s" % str(e), file=sys.stderr)


# This code is adapted from: https://github.com/outerbounds/monitorbench
class NeuronProfiler:
    def __init__(self, interval=1, monitor_batch_duration=10):
        self.error = False
        self._monitor = NeuronMonitor(
            interval=interval, duration=monitor_batch_duration
        )
        self._monitor_thread = threading.Thread(
            target=self._monitor._monitor_update_thread,  # daemon=True
        )
        self._monitor_thread.start()
        self._interval = interval

        time.sleep(1)
        
        iter = 0
        while True:
            self.hardware_info_dict = self._monitor.read_hardware_info()
            try:
                self.devices = [
                    str(i)
                    for i in range(
                        int(self.hardware_info_dict["neuron_device_count"])
                        * int(self.hardware_info_dict["neuroncore_per_device_count"])
                    )
                ]
                break
            except TypeError as e:
                time.sleep(3)
                iter += 1
                if iter > 5:
                    raise ValueError("Couldn't read hardware info from the monitor")

        self._card_comps = {"max_utilization": {}, "charts": {}, "reading_duration": {}}
        self._card_created = False

    def finish(self):
        ret = {}
        if self.error:
            return ret
        else:
            ret["devices"] = self.devices
            ret["profile"] = self._monitor.read()
            self._monitor.cleanup()
            return ret

    def _make_reading(self):
        ret = {}
        if self.error:
            return ret
        else:
            ret["devices"] = self.devices
            ret["profile"] = self._monitor.read()
            return ret

    def _update_card(self):
        if len(self.devices) == 0:
            current.card["neuron_monitor"].clear()
            current.card["neuron_monitor"].append(
                Markdown("## Neuron monitor failed: %s" % self.error)
            )
            current.card["neuron_monitor"].refresh()

            return

        while True:
            readings = self._make_reading()
            if readings is None:
                print("Neuron monitor readings are none", file=sys.stderr)
                time.sleep(self._interval)
                continue
            if len(readings["profile"]) == 0:
                time.sleep(self._interval)
                continue
            _update_utilization(readings, self._card_comps["max_utilization"])
            _update_charts(readings, self._card_comps["charts"])
            current.card["neuron_monitor"].refresh()
            time.sleep(self._interval)

    def _setup_card(self, artifact_name):
        from metaflow import current

        results = self._make_reading()
        els = current.card["neuron_monitor"]

        def _devices():
            els.append(Markdown("## Devices"))
            els.append(Artifact({"devices": self.hardware_info_dict}))

        def _utilization():
            els.append(Markdown("## Maximum utilization"))
            rows = {}
            for device_id in results["devices"]:
                rows[device_id] = {
                    "neuron": Markdown("0%"),
                    "memory": Markdown("0MB"),
                }
            _rows = [[Markdown(k)] + list(v.values()) for k, v in rows.items()]
            els.append(
                Table(
                    data=_rows, headers=["Device ID", "Max neuron core %", "Max memory"]
                )
            )
            els.append(
                Markdown(f"Detailed data saved in an artifact `{artifact_name}`")
            )
            return rows

        def _plots():
            els.append(
                Markdown("## Neuron core utilization and memory usage over time")
            )
            rows = {}
            for device_id in results["devices"]:
                neuron_plot, mem_plot, ts_range = profile_plots(
                    device_id, [], [], [], []
                )
                rows[device_id] = {
                    "neuron": VegaChart(neuron_plot),
                    "memory": VegaChart(mem_plot),
                    "reading_duration": Markdown(_get_ts_range(ts_range)),
                }
            for k, v in rows.items():
                els.append(Markdown("### Neuron Utilization for device : %s" % k))
                els.append(v["reading_duration"])
                els.append(
                    Table(
                        data=[
                            [Markdown("Neuron Utilization"), v["neuron"]],
                            [Markdown("Memory usage"), v["memory"]],
                        ]
                    )
                )
            return rows

        _devices()
        self._card_comps["max_utilization"] = _utilization()
        self._card_comps["charts"] = _plots()


class neuron_monitor:

    def __init__(
        self,
        include_artifacts=True,
        artifact_prefix="neuron_monitor_",
        interval=1,
    ):
        self.include_artifacts = include_artifacts
        self.artifact_prefix = artifact_prefix
        self.interval = interval

    def __call__(self, f):
        @wraps(f)
        def func(s):
            prof = NeuronProfiler(interval=self.interval)
            if self.include_artifacts:
                setattr(s, self.artifact_prefix + "num_neuron", len(prof.devices))

            current.card["neuron_monitor"].append(
                Markdown("# Neuron monitor for `%s`" % current.pathspec)
            )
            current.card["neuron_monitor"].append(
                Markdown(
                    "_Started at: %s_"
                    % datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S %z")
                )
            )
            prof._setup_card(self.artifact_prefix + "data")
            current.card["neuron_monitor"].refresh()
            update_thread = threading.Thread(target=prof._update_card, daemon=True)
            update_thread.start()

            try:
                f(s)
            finally:
                try:
                    results = prof.finish()
                except:
                    results = {"error": "couldn't read profiler results"}
                if self.include_artifacts:
                    setattr(s, self.artifact_prefix + "data", results)

        from metaflow import card

        return card(type="blank", id="neuron_monitor", refresh_interval=self.interval)(
            func
        )


def translate_to_vegalite(
    tstamps,
    vals,
    description,
    y_label,
    legend,
    line_color=None,
    percentage_format=False,
):
    # Preprocessing for Vega-Lite
    # Assuming tstamps is a list of datetime objects and vals is a list of values
    data = [{"tstamps": str(t), "vals": v} for t, v in zip(tstamps, vals)]

    # Base Vega-Lite spec
    vega_lite_spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": description,
        "data": {"values": data},
        "width": 600,
        "height": 400,
        "encoding": {
            "x": {"field": "tstamps", "type": "temporal", "axis": {"title": "Time"}},
            "y": {
                "field": "vals",
                "type": "quantitative",
                "axis": {
                    "title": y_label,
                    **({"format": "%"} if percentage_format else {}),
                },
            },
        },
        "layer": [
            {
                "mark": {
                    "type": "line",
                    "color": line_color if line_color else "blue",
                    "tooltip": True,
                    "description": legend,  # Adding legend as description
                },
                "encoding": {"tooltip": [{"field": "tstamps"}, {"field": "vals"}]},
            }
        ],
    }

    return vega_lite_spec


def profile_plots(device_id, ts, neuron, mem_used, mem_total):
    tstamps = [datetime.strptime(t, TS_FORMAT) for t in ts]
    neuron = [i / 100 for i in list(map(float, neuron))]
    mem = [float(used) / float(total) for used, total in zip(mem_used, mem_total)]
    time_stamp_range = ""
    if len(tstamps) > 1:
        max_time = max(tstamps).strftime(TS_FORMAT)
        min_time = min(tstamps).strftime(TS_FORMAT)
        time_stamp_range = "%s to %s" % (min_time, max_time)

    neuron_plot = translate_to_vegalite(
        tstamps,
        neuron,
        "Neuron core utilization",
        "Neuron core utilization",
        "device: %s" % device_id,
        line_color=NEURON_COLOR,
        percentage_format=True,
    )
    mem_plot = translate_to_vegalite(
        tstamps,
        mem,
        "Percentage Memory utilization",
        "Percentage Memory utilization",
        "device: %s" % device_id,
        line_color=MEM_COLOR,
        percentage_format=True,
    )
    return neuron_plot, mem_plot, time_stamp_range


if __name__ == "__main__":
    prof = NeuronProfiler(monitor_batch_duration=10)

    def _write_json_file(data, filename):
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    import time

    for i in range(15):
        time.sleep(1)
        _write_json_file(prof._monitor.read(), "neuron_monitor.json")

    print(json.dumps(prof.finish()))
