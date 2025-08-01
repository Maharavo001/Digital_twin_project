from basePLC import BasePLC
from utils import PLC2_DATA, STATE, PLC2_PROTOCOL
from utils import TANK, PUMP1, PUMP2, ATT_1, PLC1_ADDR, PLC2_ADDR, flag_attack_plc2, CONTROL
import csv
from datetime import datetime
import logging
from decimal import Decimal
import time
import signal
import sys
import threading


class PLC2(BasePLC):

    def pre_loop(self):

        print('DEBUG: plc2 enters pre_loop')

        self.local_time = 0
        self.reader = True

        self.pu1 = int(self.get(PUMP1))
        self.pu2 = int(self.get(PUMP2))

        self.saved_tank_levels = [["iteration", "timestamp", "TANK_LEVEL"]]
        path = 'plc2_saved_tank_levels_received.csv'
        self.lock = threading.Lock()
        lastPLC = True
        week_index = int(sys.argv[1])
        BasePLC.set_parameters(self, path, self.saved_tank_levels, [PUMP1, PUMP2], [self.pu1, self.pu2], self.reader,
                               self.lock, PLC2_ADDR)
        self.startup()

    def main_loop(self):
        """plc2 main loop.
            - read flow level sensors #2
            - update interval enip server
        """

        print('DEBUG: plc2 enters main_loop.')
        while True:
            control = int(self.get(CONTROL))
            if control == 0:
                try:
                    self.tank_level = Decimal(self.receive(TANK, PLC1_ADDR))
                except Exception:
                    continue

                self.local_time += 1
                self.saved_tank_levels.append([self.local_time, datetime.now(), self.tank_level])

                if flag_attack_plc2:
                    if 300 <= self.local_time <= 450:
                        self.set(ATT_1, 1)
                        time.sleep(0.1)
                        continue
                    else:
                        self.set(ATT_1, 0)

                if self.tank_level < 4:
                    self.pu1 = 1

                if self.tank_level > 6.3:
                    self.pu1 = 0

                # CONTROL PUMP2
                if self.tank_level < 1:
                    self.pu2 = 1

                if self.tank_level > 4.5:
                    self.pu2 = 0

                self.set(PUMP1, self.pu1)
                self.set(PUMP2, self.pu2)

                self.set(CONTROL, 1)  # <-- IMPORTANT: signal end of iteration
                time.sleep(0.1)
            else:
                time.sleep(0.05)  # avoid 100% CPU usage when control != 0


if __name__ == "__main__":
    plc2 = PLC2(
        name='plc2',
        state=STATE,
        protocol=PLC2_PROTOCOL,
        memory=PLC2_DATA,
        disk=PLC2_DATA)
    plc2.pre_loop()
    plc2.main_loop()

