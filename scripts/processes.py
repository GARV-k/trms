import os, signal, socket, subprocess
from scripts.experiment import ROOT

def command(module, args):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    return ['bash', str(ROOT/'scripts/container.sh'), '-m', 'torch.distributed.run',
        '--nnodes=1', '--nproc_per_node=4', '--master_addr=127.0.0.1',
        '--master_port='+str(port), '--max_restarts=0', '-m', module, *args]


def run_case(cmd, log, timeout):
    # Kill the entire launcher/worker group on timeout before the next candidate.
    with log.open('w') as stream:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            stream.write('\nPROFILE TIMEOUT: worker group terminated.\n')
            return 124

