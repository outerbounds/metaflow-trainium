from metaflow.cards import Markdown, Table, VegaChart
from functools import wraps
import threading
from datetime import datetime
from metaflow import current
from metaflow.cards import Table, Markdown, VegaChart, Image
import time
import shutil

import re
import os
import uuid
import json
import sys
from tempfile import TemporaryFile, TemporaryDirectory
from subprocess import check_output, Popen
from datetime import datetime, timedelta
from functools import wraps
from collections import namedtuple

# Card plot styles
MEM_COLOR = "#0c64d6"
GPU_COLOR = "#ff69b4"

NVIDIA_TS_FORMAT = "%Y/%m/%d %H:%M:%S"


DRIVER_VER = re.compile(b"Driver Version: (.+?) ")
CUDA_VER = re.compile(b"CUDA Version:(.*) ")

MONITOR_FIELDS = [
    "timestamp",
    "gpu_utilization",
    "memory_used",
    "memory_total",
]

MONITOR = """nvidia-smi --query-gpu=pci.bus_id,timestamp,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits -l {interval};"""
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
    This class is responsible for managing the nvidia SMI subprocesses
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
        return datetime.strptime(ts, NVIDIA_TS_FORMAT)
    except ValueError:
        return None


class GPUMonitor:
    """
    The `GPUMonitor` class is designed to monitor GPU usage.

    When an instance of `GPUMonitor` is created, it initializes with a specified `interval` and `duration`.
    The `duration` is the timeperiod it will run the NVIDIA SMI command for and the `interval` is the timeperiod between each reading.
    The class exposes a `_monitor_update_thread` method which runs as a background thread that continuously updates the GPU usage readings.
    It will keep running unitl the `_finished` flag is set to `True`.

    The class will statefully manage the the spawned NVIDI-SMI processes.
    It will start a new NVIDI-SMI process after the current one has ran for the specified `duration`.
    At a time this class will only maintain readings for the `_current_process` and will have all the aggregated
    readings for the past processes stored in the `_past_readings` dictionary.
    When a process finishes completion, the readings are appended to the `_past_readings` dictionary and a new process is started.

    If the caller of this class wishes to read the GPU usage, they can call the `read` method which will return the readings in a dictionary format.
    The `read` method will aggregate the readings from the `_current_readings` and `_past_readings`.
    """

    _started_processes = []

    _current_process: ProcessUUID = None

    _current_readings = {}

    _past_readings = {}

    def __init__(self, interval=1, duration=300) -> None:
        self._tempdir = TemporaryDirectory(prefix="gpu_card_monitor", dir="./")
        self._interval = interval
        self._duration = duration
        self._finished = False

    @property
    def _current_file(self):
        if self._current_process is None:
            return None
        return os.path.join(self._tempdir.name, self._current_process.uuid + ".csv")

    def get_file_name(self, uuid):
        return os.path.join(self._tempdir.name, uuid + ".csv")

    def create_new_monitor(self):
        uuid = _get_uuid(self._duration)
        file = open(self.get_file_name(uuid.uuid), "w")
        cmd = MONITOR.format(interval=self._interval, time_duration=self._duration)
        AsyncProcessManager.spawn(uuid.uuid, ["bash", "-c", cmd], file)
        self._started_processes.append(uuid)
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
        all_readings = []
        if self._current_file is None:
            return None
        # Extract everything from the CVS File and store it in a list of dictionaries
        all_fields = ["gpu_id"] + MONITOR_FIELDS
        with open(self._current_file, "r") as _monitor_out:
            for line in _monitor_out.readlines():
                data = {}
                fields = [f.strip() for f in line.split(",")]
                if len(fields) == len(all_fields):
                    # strip subsecond resolution from timestamps that doesn't align across devices
                    for idx, _f in enumerate(all_fields):
                        data[_f] = fields[idx]
                    all_readings.append(data)
                else:
                    # expect that the last line may be truncated
                    break

        # Convert to dictionary format
        devdata = {}
        for reading in all_readings:
            gpu_id = reading["gpu_id"]
            if "timestamp" not in reading:
                continue
            if _parse_timestamp(reading["timestamp"]) is None:
                continue
            reading["timestamp"] = reading["timestamp"].split(".")[0]
            if gpu_id not in devdata:
                devdata[gpu_id] = {}

            for i, field in enumerate(MONITOR_FIELDS):
                if field not in devdata[gpu_id]:
                    devdata[gpu_id][field] = []
                devdata[gpu_id][field].append(reading[field])
        return devdata

    def _update_readings(self):
        """
        Core update function that checks if the current process has ended and if so, it will create a new monitor
        otherwise sets the current readings to the readings from the monitor file
        """
        if self.current_process_has_ended() or not self.current_process_is_running():
            self._update_past_readings()
            self.clear_current_monitor()
            self.create_new_monitor()
            # Sleep for 1 seconds to allow the new process to start and we can make a reading
            time.sleep(1)

        readings = self._read_monitor()
        if readings is None:
            return
        self._current_readings = readings

    @staticmethod
    def _make_full_reading(current, past):
        if current is None:
            return past
        for gpu_id in current:
            if gpu_id not in past:
                past[gpu_id] = {}
            for field in MONITOR_FIELDS:
                if field not in past[gpu_id]:
                    past[gpu_id][field] = []
                past[gpu_id][field].extend(current[gpu_id][field])
        return past

    def read(self):
        return self._make_full_reading(
            self._current_readings, json.loads(json.dumps(self._past_readings))
        )

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
                "Device %s not found in the GPU card layout. Skipping..." % device,
                file=sys.stderr,
            )
            continue
        md_dict[device]["gpu"].update(
            "%2.1f%%" % max(map(float, data["gpu_utilization"]))
        )
        md_dict[device]["memory"].update("%dMB" % max(map(float, data["memory_used"])))


def _update_charts(results, md_dict):
    for device, data in results["profile"].items():
        try:
            if device not in md_dict:
                continue
            gpu_plot, mem_plot, ts_range = profile_plots(
                device,
                data["timestamp"],
                data["gpu_utilization"],
                data["memory_used"],
                data["memory_total"],
            )
            md_dict[device]["gpu"].update(gpu_plot)
            md_dict[device]["memory"].update(mem_plot)
            md_dict[device]["reading_duration"].update(_get_ts_range(ts_range))
        except ValueError as e:
            # This is thrown when the date is unparsable. We can just safely ignore this.
            print("ValueError: Could not parse date \n%s" % str(e), file=sys.stderr)


# This code is adapted from: https://github.com/outerbounds/monitorbench
class GPUProfiler:
    def __init__(self, interval=1, monitor_batch_duration=200):
        self.driver_ver, self.cuda_ver, self.error = self._read_versions()
        (
            self.interconnect_data,
            self.interconnect_legend,
        ) = self._read_multi_gpu_interconnect()
        if self.error:
            self.devices = []
            return
        else:
            self.devices = self._read_devices()
            self._monitor = GPUMonitor(
                interval=interval, duration=monitor_batch_duration
            )
            self._monitor_thread = threading.Thread(
                target=self._monitor._monitor_update_thread, daemon=True
            )
            self._monitor_thread.start()
            self._interval = interval

        self._card_comps = {"max_utilization": {}, "charts": {}, "reading_duration": {}}
        self._card_created = False

    def finish(self):
        ret = {
            "error": self.error,
            "cuda_version": self.cuda_ver,
            "driver_version": self.driver_ver,
        }
        if self.error:
            return ret
        else:
            ret["devices"] = self.devices
            ret["profile"] = self._monitor.read()
            ret["interconnect"] = {
                "data": self.interconnect_data,
                "legend": self.interconnect_legend,
            }
            self._monitor.cleanup()
            return ret

    def _make_reading(self):
        ret = {
            "error": self.error,
            "cuda_version": self.cuda_ver,
            "driver_version": self.driver_ver,
        }
        if self.error:
            return ret
        else:
            ret["devices"] = self.devices
            ret["profile"] = self._monitor.read()
            ret["interconnect"] = {
                "data": self.interconnect_data,
                "legend": self.interconnect_legend,
            }
            return ret

    def _update_card(self):
        if len(self.devices) == 0:
            current.card["gpu_profile"].clear()
            current.card["gpu_profile"].append(
                Markdown("## GPU profile failed: %s" % self.error)
            )
            current.card["gpu_profile"].refresh()

            return

        while True:
            readings = self._make_reading()
            if readings is None:
                print("GPU Profiler readings are none", file=sys.stderr)
                time.sleep(self._interval)
                continue
            _update_utilization(readings, self._card_comps["max_utilization"])
            _update_charts(readings, self._card_comps["charts"])
            current.card["gpu_profile"].refresh()
            time.sleep(self._interval)

    def _setup_card(self, artifact_name):
        from metaflow import current

        results = self._make_reading()
        els = current.card["gpu_profile"]

        def _drivers():
            els.append(Markdown("## Drivers"))
            els.append(
                Table(
                    [[results["cuda_version"], results["driver_version"]]],
                    headers=["NVidia driver version", "CUDA version"],
                )
            )

        def _devices():
            els.append(Markdown("## Devices"))
            rows = [
                [d["device_id"], d["name"], d["memory"]] for d in results["devices"]
            ]
            els.append(Table(rows, headers=["Device ID", "Device type", "GPU memory"]))

        def _interconnect():
            if results["interconnect"]["data"] and results["interconnect"]["legend"]:
                els.append(Markdown("## Interconnect"))
                interconnect_data = results["interconnect"]["data"]
                rows = list(interconnect_data.values())
                rows = [list(transpose_row) for transpose_row in list(zip(*rows))]
                els.append(Table(rows, headers=list(interconnect_data.keys())))
                els.append(Markdown("#### Legend"))
                els.append(
                    Table(
                        [list(results["interconnect"]["legend"].values())],
                        headers=list(results["interconnect"]["legend"].keys()),
                    )
                )

        def _utilization():
            els.append(Markdown("## Maximum utilization"))
            rows = {}
            for d in results["devices"]:
                rows[d["device_id"]] = {
                    "gpu": Markdown("0%"),
                    "memory": Markdown("0MB"),
                }
            _rows = [[Markdown(k)] + list(v.values()) for k, v in rows.items()]
            els.append(
                Table(data=_rows, headers=["Device ID", "Max GPU %", "Max memory"])
            )
            els.append(
                Markdown(f"Detailed data saved in an artifact `{artifact_name}`")
            )
            return rows

        def _plots():
            els.append(Markdown("## GPU utilization and memory usage over time"))

            rows = {}
            for d in results["devices"]:
                gpu_plot, mem_plot, ts_range = profile_plots(
                    d["device_id"], [], [], [], []
                )
                rows[d["device_id"]] = {
                    "gpu": VegaChart(gpu_plot),
                    "memory": VegaChart(mem_plot),
                    "reading_duration": Markdown(_get_ts_range(ts_range)),
                }
            for k, v in rows.items():
                els.append(Markdown("### GPU Utilization for device : %s" % k))
                els.append(v["reading_duration"])
                els.append(
                    Table(
                        data=[
                            [Markdown("GPU Utilization"), v["gpu"]],
                            [Markdown("Memory usage"), v["memory"]],
                        ]
                    )
                )
            return rows

        _drivers()
        _devices()
        _interconnect()
        self._card_comps["max_utilization"] = _utilization()
        self._card_comps["charts"] = _plots()

    def _read_versions(self):
        def parse(r, s):
            return r.search(s).group(1).strip().decode("utf-8")

        try:
            out = check_output(["nvidia-smi"])
            return parse(DRIVER_VER, out), parse(CUDA_VER, out), None
        except FileNotFoundError:
            return None, None, "nvidia-smi not found"
        except AttributeError:
            return None, None, "nvidia-smi output is unexpected"
        except:
            return None, None, "nvidia-smi error"

    def _read_devices(self):
        out = check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,pci.bus_id,memory.total",
                "--format=csv,noheader",
            ]
        )
        return [
            dict(
                zip(("name", "device_id", "memory"), (x.strip() for x in l.split(",")))
            )
            for l in out.decode("utf-8").splitlines()
        ]

    def _read_multi_gpu_interconnect(self):
        """
        parse output of `nvidia-smi tomo -m`, such as this sample:

            GPU0    GPU1    CPU Affinity    NUMA Affinity
            GPU0     X      NV2     0-23            N/A
            GPU1    NV2      X      0-23            N/A

        returns two dictionaries describing multi-GPU topology:
            data: {index: [GPU0, GPU1, ...], GPU0: [X, NV2, ...], GPU1: [NV2, X, ...], ...}
            legend_items: {X: 'Same PCI', NV2: 'NVLink 2', ...}
        """
        try:
            import re

            ansi_escape = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")

            out = check_output(["nvidia-smi", "topo", "-m"])
            rows = out.decode("utf-8").split("\n")

            header = ansi_escape.sub("", rows[0]).split("\t")[1:]
            data = {}
            data["index"] = []
            data |= {k: [] for k in header}

            for i, row in enumerate(rows[1:]):
                row = ansi_escape.sub("", row).split()
                if len(row) == 0:
                    continue
                if row[0].startswith("GPU"):
                    data["index"].append(row[0])
                    for key, val in zip(header, row[1:]):
                        data[key].append(val)
                elif row[0].startswith("Legend"):
                    break

            legend_items = {}
            for legend_row in rows[i:]:
                if legend_row == "" or legend_row.startswith("Legend"):
                    continue
                res = legend_row.strip().split(" = ")
                legend_items[res[0].strip()] = res[1].strip()

            return data, legend_items

        except:
            return None, None


class gpu_profile:
    def __init__(
        self,
        include_artifacts=True,
        artifact_prefix="gpu_profile_",
        interval=1,
    ):
        self.include_artifacts = include_artifacts
        self.artifact_prefix = artifact_prefix
        self.interval = interval

    def __call__(self, f):
        @wraps(f)
        def func(s):
            prof = GPUProfiler(interval=self.interval)
            if self.include_artifacts:
                setattr(s, self.artifact_prefix + "num_gpus", len(prof.devices))

            current.card["gpu_profile"].append(
                Markdown("# GPU profile for `%s`" % current.pathspec)
            )
            current.card["gpu_profile"].append(
                Markdown(
                    "_Started at: %s_"
                    % datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S %z")
                )
            )
            prof._setup_card(self.artifact_prefix + "data")
            current.card["gpu_profile"].refresh()
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

        return card(type="blank", id="gpu_profile", refresh_interval=self.interval)(
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


def profile_plots(device_id, ts, gpu, mem_used, mem_total):
    tstamps = [datetime.strptime(t, NVIDIA_TS_FORMAT) for t in ts]
    gpu = [i / 100 for i in list(map(float, gpu))]
    mem = [float(used) / float(total) for used, total in zip(mem_used, mem_total)]
    time_stamp_range = ""
    if len(tstamps) > 1:
        max_time = max(tstamps).strftime(NVIDIA_TS_FORMAT)
        min_time = min(tstamps).strftime(NVIDIA_TS_FORMAT)
        time_stamp_range = "%s to %s" % (min_time, max_time)

    gpu_plot = translate_to_vegalite(
        tstamps,
        gpu,
        "GPU utilization",
        "GPU utilization",
        "device: %s" % device_id,
        line_color=GPU_COLOR,
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
    return gpu_plot, mem_plot, time_stamp_range


if __name__ == "__main__":
    prof = GPUProfiler(monitor_batch_duration=10)

    def _write_json_file(data, filename):
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    import time

    for i in range(15):
        time.sleep(1)
        _write_json_file(prof._monitor.read(), "gpu_profile.json")

    print(json.dumps(prof.finish()))
