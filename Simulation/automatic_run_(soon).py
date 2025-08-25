
# -*- coding: utf-8 -*-

from mininet.net import Mininet
from mininet.cli import CLI
from minicps.mcps import MiniCPS
from topo import ScadaTopo
import sys
import time
import shlex
import subprocess
import signal

# Passez à 1 pour lancer automatiquement sans CLI
automatic = 1

class Minitown(MiniCPS):
    """ Script to run the Minitown SCADA topology """

    def __init__(self, name, net):
        # Gestion interruption Ctrl-C
        signal.signal(signal.SIGINT, self.interrupt)
        signal.signal(signal.SIGTERM, self.interrupt)

        self.net = net
        net.start()

        # Activation du forwarding sur le routeur
        r0 = net.get('r0')
        r0.cmd('sysctl net.ipv4.ip_forward=1')

        # Récupération des hôtes
        plc1  = net.get('plc1')
        plc2  = net.get('plc2')
        scada = net.get('scada')

        # Ajout des routes pour les deux réseaux
        plc1.cmd('route add -net 192.168.2.0/24 gw 192.168.1.254')
        plc2.cmd('route add -net 192.168.2.0/24 gw 192.168.1.254')
        scada.cmd('route add default gw 192.168.2.254')

        # Index de la semaine (argument)
        if len(sys.argv) > 1:
            self.week_index = sys.argv[1]
        else:
            self.week_index = '0'

        if automatic:
            self.automatic_start()
        else:
            CLI(net)

        net.stop()

    def interrupt(self, sig, frame):
        self.finish()
        sys.exit(0)

    def automatic_start(self):
        plc1  = self.net.get('plc1')
        plc2  = self.net.get('plc2')
        scada = self.net.get('scada')
        plant = self.net.get('plant')

        self.create_log_files()

        # Ouverture des fichiers de log
        try:
            plc1_output     = open("output/plc1.log", 'w')
            plc2_output     = open("output/plc2.log", 'w')
            scada_output    = open("output/scada.log", 'w')
            physical_output = open("output/physical.log", 'w')
        except IOError as e:
            print(("Error creating output files: {}".format(e)))
            return

        # Lancement des processus PLC et SCADA
        for name, host, output in [
            ('plc1',  plc1,  plc1_output),
            ('plc2',  plc2,  plc2_output),
            ('scada', scada, scada_output)
        ]:
            venv_python = "/home/zayd/Desktop/Digital_twin_project/digitaltwin/bin/python"
            cmd_str = f"{venv_python} automatic_plc.py -n {name} -w {self.week_index}"

            print(("Running {} process with command: {}".format(name, cmd_str)))
            cmd = shlex.split(cmd_str)
            process = host.popen(cmd, stderr=sys.stdout, stdout=output)
            setattr(self, "{}_process".format(name), process)

        # Lancement de la simulation plant
        sim_cmd_str = f"{venv_python} automatic_plant.py minitown_config.yaml {self.week_index}"

        print(("Running plant simulation with command: {}".format(sim_cmd_str)))
        sim_cmd = shlex.split(sim_cmd_str)
        self.simulation = plant.popen(sim_cmd, stderr=sys.stdout, stdout=physical_output)

        print("[*] Launched all processes, simulating...")
        while self.simulation.poll() is None:
            time.sleep(1)
        self.finish()

    def create_log_files(self):
        try:
            subprocess.call(["mkdir", "-p", "output"])
        except Exception as e:
            print(("Error creating output directory: {}".format(e)))

    def end_process(self, proc):
        if proc and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait()
            except Exception:
                pass
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait()
                except Exception:
                    pass
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass

    def finish(self):
        print("[*] Simulation finished")
        # Arrêt propre des processus
        for attr in ['scada_process', 'plc2_process', 'plc1_process', 'simulation']:
            if hasattr(self, attr):
                self.end_process(getattr(self, attr))

        # Nettoyage final
        try:
            subprocess.call(shlex.split("./kill_cppo.sh"))
        except Exception:
            pass

        if hasattr(self, 'net'):
            self.net.stop()
        sys.exit(0)


if __name__ == "__main__":
    topo = ScadaTopo()
    net  = Mininet(topo=topo)
    Minitown(name='minitown', net=net)

