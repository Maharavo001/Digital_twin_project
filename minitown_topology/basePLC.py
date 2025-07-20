from minicps.devices import PLC
import csv
import signal
import sys
import threading
import shlex
import subprocess
import time

class BasePLC(PLC):
    def send_system_state(self):
        """
        This method sends the current values of all tags to the SCADA server
        or another client at regular intervals.
        """
        while self.reader:
            values = []
            for tag in self.tags:
                with self.lock:
                    try:
                        # Retrieve the current value of the tag
                        values.append(self.get(tag))
                    except Exception:
                        print "Exception trying to get the tag"
                        time.sleep(0.05)
                        continue
            # Send all tags and their corresponding values
            self.send_multiple(self.tags, values, self.send_address)
            time.sleep(0.05)

    def send_multiple(self, tags, values, address):
        """
        Sends each tag-value pair individually to the specified address.
        """
        for tag, value in zip(tags, values):
            try:
                # Convert tag to tuple as required by MiniCPS send method
                tag_tuple = (tag,) if isinstance(tag, str) else tag
                self.send(tag_tuple, value, address)
            except Exception as e:
                print "Exception sending tag {}: {}".format(tag, e)

    def receive_multiple(self, tags, address):
        """
        Receives values for each tag individually from the specified address.
        Returns a list of values corresponding to the tags.
        """
        values = []
        for tag in tags:
            try:
                tag_tuple = (tag,) if isinstance(tag, str) else tag
                value = self.receive(tag_tuple, address)
                values.append(value)
            except Exception as e:
                print "Exception receiving tag {}: {}".format(tag, e)
                values.append(None)
        return values

    def set_parameters(self, path, result_list, tags, values, reader, lock, send_address, send_port=44818, lastPLC=False, week_index=0, isScada=False):
        """
        Initializes the parameters required by the PLC instance.
        """
        self.result_list = result_list
        self.path = path
        self.tags = tags
        self.values = values
        self.reader = reader
        self.lock = lock
        self.send_address = send_address
        self.send_port = send_port
        self.lastPLC = lastPLC
        self.week_index = week_index
        self.isScada = isScada

    def write_output(self):
        """
        Writes the result_list to a CSV file in the output folder.
        """
        with open('output/' + self.path, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(self.result_list)

    def sigint_handler(self, sig, frame):
        """
        Handles graceful shutdown when receiving termination signals.
        Saves results and triggers any final file movements if needed.
        """
        print 'DEBUG PLC shutdown'
        self.reader = False
        self.write_output()
        if self.lastPLC:
            self.move_files()
        sys.exit(0)

    def move_files(self):
        """
        Executes an external shell script to copy the output files.
        """
        cmd = shlex.split("./copy_output.sh " + str(self.week_index))
        subprocess.call(cmd)

    def startup(self):
        """
        Sets up signal handlers and starts the data-sending thread.
        """
        signal.signal(signal.SIGINT, self.sigint_handler)
        signal.signal(signal.SIGTERM, self.sigint_handler)
        if not self.isScada:
            threading.Thread(target=self.send_system_state).start()
