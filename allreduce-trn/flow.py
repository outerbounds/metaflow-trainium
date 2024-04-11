from metaflow import FlowSpec, step, batch, parallel

N_NODES = 2
N_TRAINIUM = 16


class TrainiumAllReduce(FlowSpec):
    @step
    def start(self):
        print("Starting...")
        self.next(self.make_instance, num_parallel=N_NODES)

    @parallel
    @batch(
        inferentia=N_TRAINIUM,
        efa=8,
        cpu=96,
        memory=500000,
        image="public.ecr.aws/outerbounds/trainium:latest",
        queue="trn1-batch-trn1_32xl_batch_job_queue",
    )
    @step
    def make_instance(self):
        import subprocess

        subprocess.run(["./allreduce.sh"])
        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    TrainiumAllReduce()
