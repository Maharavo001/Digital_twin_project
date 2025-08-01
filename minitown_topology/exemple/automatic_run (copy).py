from mininet.net import Mininet
from mininet.cli import CLI
from minicps.mcps import MiniCPS
from topo import ScadaTopo
import sys
import time
import shlex
import subprocess
import signal

automatic = 1

class Minitown(MiniCPS):
    """ Script to run the Minitown SCADA topology """

    def __init__(self, name, net):
        signal.signal(signal.SIGINT, self.interrupt)
        signal.signal(signal.SIGTERM, self.interrupt)
        net.start()

        if len(sys.argv) > 1:
            self.week_index = sys.argv[1]
        else:
            self.week_index = str(0)

        r0 = net.get('r0')
        # Pre experiment configuration, prepare routing path
        r0.cmd('sysctl net.ipv4.ip_forward=1')

        if automatic:
            self.automatic_start()
        else:
            CLI(net)
        net.stop()

    def interrupt(self, sig, frame):
        self.finish()
        sys.exit(0)

    def automatic_start(self):

        plc1 = net.get('plc1')
        plc2 = net.get('plc2')
        scada = net.get('scada')

        self.create_log_files()

        plc1_output = open("output/plc1.log", 'r+')
        plc2_output = open("output/plc2.log", 'r+')
        scada_output = open("output/scada.log", 'r+')
        physical_output = open("output/physical.log", 'r+')

        string_command = "python automatic_plc.py -n plc1 -w " + str(self.week_index)
        cmd = shlex.split(string_command)
        print("Running plc process with command " + string_command)
        self.plc1_process = plc1.popen(cmd, stderr=sys.stdout, stdout=plc1_output )

        string_command = "python automatic_plc.py -n plc2 -w " + str(self.week_index)
        cmd = shlex.split(string_command)
        print("Running plc process with command " + string_command)
        self.plc2_process = plc2.popen(cmd, stderr=sys.stdout, stdout=plc2_output )

        string_command = "python automatic_plc.py -n scada -w " + str(self.week_index)
        cmd = shlex.split(string_command)
        print("Running plc process with command " + string_command)
        self.scada_process = scada.popen(cmd, stderr=sys.stdout, stdout=scada_output )

        print("[*] Launched the PLCs and SCADA process, launching simulation...")
        plant = net.get('plant')
        simulation_cmd = shlex.split("python automatic_plant.py minitown_config.yaml " + str(self.week_index))
        self.simulation = plant.popen(simulation_cmd, stderr=sys.stdout, stdout=physical_output)

        print("[] Simulating...")
        while self.simulation.poll() is None:
            pass
        self.finish()

    def create_log_files(self):
        subprocess.call("./create_log_files.sh")

    def end_process(self, a_process):
        import signal

        if a_process and a_process.poll() is None:
            try:
                a_process.send_signal(signal.SIGINT)
                a_process.wait()
            except Exception:
                pass

            if a_process.poll() is None:
                try:
                    a_process.terminate()
                    a_process.wait()
                except Exception:
                    pass

            if a_process.poll() is None:
                try:
                    a_process.kill()
                except Exception:
                    pass

    def finish(self):
        print("[*] Simulation finished")
        self.end_process(self.scada_process)
        self.end_process(self.plc2_process)
        self.end_process(self.plc1_process)

        if hasattr(self, 'simulation') and self.simulation:
            self.end_process(self.simulation)

        cmd = shlex.split("./kill_cppo.sh")
        subprocess.call(cmd)

        net.stop()
        sys.exit(0)


if __name__ == "__main__":
    topo = ScadaTopo()
    net = Mininet(topo=topo)
    minitown_cps = Minitown(name='minitown', net=net)

