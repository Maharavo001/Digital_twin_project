"""
sqlite and enip use name (string) and pid (int) has key and the state stores
values as strings.

Actuator tags are redundant, we will use only the XXX_XXX_OPEN tag ignoring
the XXX_XXX_CLOSE with the following convention:
    - 0 = error
    - 1 = off
    - 2 = on

sqlite uses float keyword and cpppo use REAL keyword.
"""

from minicps.utils import build_debug_logger

thesis = build_debug_logger(
    name=__name__,
    bytes_per_file=10000,
    rotating_files=2,
    lformat='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    ldir='logs/',
    suffix='')

# SPHINX_SWAT_TUTORIAL PROCESS UTILS(

SCADA_PERIOD_SEC = 2.0

# ImageTk
DISPLAYED_SAMPLES = 14

PLC_PERIOD_SEC = 0.40  # plc update rate in seconds
PLC_PERIOD_HOURS = PLC_PERIOD_SEC / 3600.0
PLC_SAMPLES = 1000

PP_RESCALING_HOURS = 100
PP_PERIOD_SEC = 0.20  # physical process update rate in seconds
PP_PERIOD_HOURS = (PP_PERIOD_SEC / 3600.0) * PP_RESCALING_HOURS
PP_SAMPLES = int(PLC_PERIOD_SEC / PP_PERIOD_SEC) * PLC_SAMPLES

TANK = ('TANK', 1)
PUMP1 = ('PUMP1', 1)
PUMP2 = ('PUMP2', 1)
ATT_1 = ('ATT_1', 1)
ATT_2 = ('ATT_2', 1)
TIME = ('TIME', 1)
CONTROL = ('CONTROL', 1)

flag_attack_plc1 = 0
flag_attack_plc2 = 0
flag_attack_communication_plc1_scada = 0
flag_attack_communication_plc1_plc2 = 0
flag_attack_dos_plc2 = 0

iteration = 0  # global variable

# topo {{{1
IP = {
    'plc1': '192.168.1.10',
    'plc2': '192.168.1.20',
    'scada': '192.168.2.30',
    'attacker': '192.168.1.77',
    'attacker2': '192.168.2.77',
}

NETMASK = '/24'

MAC = {
    'plc1': '00:1D:9C:C7:B0:70',
    'plc2': '00:1D:9C:C8:BC:46',
    'scada': '64:00:6A:70:86:D0',
    'attacker': 'AA:AA:AA:AA:AA:AA',
    'attacker2': 'BB:BB:BB:BB:BB:BB',
}

PLC1_ADDR = IP['plc1']
PLC2_ADDR = IP['plc2']
SCADA_ADDR = IP['scada']

PLC1_DATA = {
    'TODO': 'TODO',
}
PLC2_DATA = {
    'TODO': 'TODO',
}

SCADA_DATA = {
    'TODO': 'TODO',
}

# Adding plc1------------------------------------------------
PLC1_TAGS = (
    ('TANK', 1, 'REAL'),
    ('PUMP1', 1, 'REAL'),
    ('PUMP2', 1, 'REAL'),
    ('ATT_1', 1, 'REAL'),
    ('ATT_2', 1, 'REAL'),
    ('TIME', 1, 'INT')
)

PLC1_SERVER = {
    'address': PLC1_ADDR,
    'tags': PLC1_TAGS
}
PLC1_PROTOCOL = {
    'name': 'enip',
    'mode': 1,
    'server': PLC1_SERVER
}
# Adding plc2-------------------------------------------
PLC2_TAGS = (
    ('TANK', 1, 'REAL'),
    ('PUMP1', 1, 'REAL'),
    ('PUMP2', 1, 'REAL'),
    ('ATT_1', 1, 'REAL'),
    ('ATT_2', 1, 'REAL'),
    ('TIME', 1, 'INT'),
    ('CONTROL', 1, 'INT')
)

PLC2_SERVER = {
    'address': PLC2_ADDR,
    'tags': PLC2_TAGS
}
PLC2_PROTOCOL = {
    'name': 'enip',
    'mode': 1,
    'server': PLC2_SERVER
}

# Adding scada--------------------------------------------
SCADA_TAGS = (
    ('TANK', 1, 'REAL'),
    ('PUMP1', 1, 'REAL'),
    ('PUMP2', 1, 'REAL'),
    ('ATT_1', 1, 'REAL'),
    ('ATT_2', 1, 'REAL'),
    ('TIME', 1, 'INT')
)

SCADA_SERVER = {
    'address': SCADA_ADDR,
    'tags': SCADA_TAGS
}

SCADA_PROTOCOL = {
    'name': 'enip',
    'mode': 1,
    'server': SCADA_SERVER
}

# Adding Attacker-----------------------------------------

ATT_ADDR = IP['attacker']

ATT_TAGS = (
    ('TANK', 1, 'REAL'),
    ('PUMP1', 1, 'REAL'),
    ('PUMP2', 1, 'REAL'),
    ('ATT_1', 1, 'REAL'),
    ('ATT_2', 1, 'REAL')
)

ATT_SERVER = {
    'address': ATT_ADDR,
    'tags': ATT_TAGS
}

ATT_PROTOCOL = {
    'name': 'enip',
    'mode': 1,
    'server': ATT_SERVER
}

# Adding Attacker_SCADA-----------------------------------------

ATT2_ADDR = IP['attacker2']

ATT2_TAGS = (
    ('TANK', 1, 'REAL'),
    ('PUMP1', 1, 'REAL'),
    ('PUMP2', 1, 'REAL'),
    ('ATT_1', 1, 'REAL'),
    ('ATT_2', 1, 'REAL')
)

ATT2_SERVER = {
    'address': ATT2_ADDR,
    'tags': ATT2_TAGS
}

ATT2_PROTOCOL = {
    'name': 'enip',
    'mode': 1,
    'server': ATT2_SERVER
}

# ---------------------------------------------------------
# SPHINX_SWAT_TUTORIAL STATE(
PATH = 'minitown_db.sqlite'
NAME = 'minitown'

STATE = {
    'name': NAME,
    'path': PATH
}
# SPHINX_SWAT_TUTORIAL STATE)

SCHEMA = """
CREATE TABLE minitown (
    name              TEXT NOT NULL,
    pid               INTEGER NOT NULL,
    value             TEXT,
    PRIMARY KEY (name, pid)
);
"""

SCHEMA_INIT = """
    INSERT INTO minitown VALUES ('V_STS', 1, '0');
    INSERT INTO minitown VALUES ('TANK', 1, '3.0');
    INSERT INTO minitown VALUES ('PUMP1', 1, '0');
    INSERT INTO minitown VALUES ('PUMP2', 1, '1');  
    INSERT INTO minitown VALUES ('ATT_1', 1, '0');  
    INSERT INTO minitown VALUES ('ATT_2', 1, '0');
    INSERT INTO minitown VALUES ('TIME', 1, '0');  
    INSERT INTO minitown VALUES ('CONTROL', 1, '1');
"""

