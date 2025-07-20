from mininet.node import Node
from mininet.topo import Topo
from utils import IP, NETMASK

class LinuxRouter(Node):
    """
    A node with IP forwarding enabled
    """
    def config(self, **params):
        super(LinuxRouter, self).config(**params)
        # Enable forwarding on the router
        self.cmd('sysctl net.ipv4.ip_forward=1')
    
    def terminate(self):
        self.cmd('sysctl net.ipv4.ip_forward=0')

class ScadaTopo(Topo):
    """
    SCADA topology
    """
    def build(self):
        # Add router
        fieldIP = '192.168.1.254/24'  # IP Address for r0-eth1
        fieldGateway = '192.168.1.254'  # Gateway IP without netmask
        
        # ---------------- FIELD NETWORK ----------------------  #
        router = self.addNode('r0', cls=LinuxRouter, ip=fieldIP)
        
        # Add switch of supervisory network
        s1 = self.addSwitch('s1')
        self.addLink(s1, router, intfName2='r0-eth1', params2={'ip': fieldIP})
        
        # Add plant host (not connected to any switch in original)
        plant = self.addHost('plant')
        
        # Add PLC hosts with proper IP configuration
        plc1 = self.addHost('plc1', ip=IP['plc1'] + NETMASK, defaultRoute=fieldGateway)
        plc2 = self.addHost('plc2', ip=IP['plc2'] + NETMASK, defaultRoute=fieldGateway)
        attacker = self.addHost('attacker', ip=IP['attacker'] + NETMASK, defaultRoute=fieldGateway)
        
        # Connect hosts to switch
        self.addLink(s1, plc1)
        self.addLink(s1, plc2)
        self.addLink(s1, attacker)
        
        # ---------------- SUPERVISORY NETWORK --------------  #
        supervisoryIP = '192.168.2.254/24'
        supervisoryGateway = '192.168.2.254'  # Gateway IP without netmask
        
        s2 = self.addSwitch('s2')
        self.addLink(s2, router, intfName2='r0-eth2', params2={'ip': supervisoryIP})
        
        # Add SCADA and attacker2 hosts
        scada = self.addHost('scada', ip=IP['scada'] + NETMASK, defaultRoute=supervisoryGateway)
        attacker2 = self.addHost('attacker2', ip=IP['attacker2'] + NETMASK, defaultRoute=supervisoryGateway)
        
        # Connect hosts to supervisory switch
        self.addLink(s2, scada)
        self.addLink(s2, attacker2)
