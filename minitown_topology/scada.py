from basePLC import BasePLC
from utils import SCADA_PROTOCOL, STATE, SCADA_DATA
from utils import PLC1_ADDR, PLC2_ADDR, TANK, PUMP1, PUMP2

import time
from datetime import datetime
from decimal import Decimal


class SCADAServer(BasePLC):

    def pre_loop(self, sleep=0.5):
        """scada pre loop.
            - sleep
        """
	time.sleep(3)
        self.saved_tank_levels = [["timestamp", "TANK_LEVEL", "PUMP1", "PUMP2"]]
        self.plc1_tags = [TANK]
        self.plc2_tags = [PUMP1, PUMP2]

        path = 'scada_saved_tank_levels_received.csv'

        isScada = True

        # Used in handling of sigint and sigterm signals, also sets the parameters to save the system state variable values into a persistent file
        BasePLC.set_parameters(self, path, self.saved_tank_levels, None, None, None, None, None, False, 0, isScada)
        self.startup()


    def main_loop(self):
        """scada main loop."""

        print("DEBUG: scada main loop")
        while True:
            try:
                plc1_values = self.receive_multiple(self.plc1_tags, PLC1_ADDR)
                plc2_values = self.receive_multiple(self.plc2_tags, PLC2_ADDR)
                results = [datetime.now()]
                results.extend(plc1_values)
                results.extend(plc2_values)
                self.saved_tank_levels.append(results)
                time.sleep(0.3)
            except Exception, msg:
                print (msg)
                continue

if __name__ == "__main__":

    scada = SCADAServer(
        name='scada',
        state=STATE,
        protocol=SCADA_PROTOCOL,
        memory=SCADA_DATA,
        disk=SCADA_DATA,
        )
    scada.pre_loop()
    scada.main_loop()
