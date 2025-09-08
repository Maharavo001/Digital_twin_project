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
        
        # ---------------- FIELD NETWORK ----------------------  #
        router = self.addNode('r0', cls=LinuxRouter, ip=fieldIP)
        
        # Add switch of field network
        s1 = self.addSwitch('s1')
        self.addLink(s1, router, intfName2='r0-eth1', params2={'ip': fieldIP})
        
        # Add plant host (not connected to any switch in original)
        plant = self.addHost('plant')
        
        # Add PLC hosts with proper IP configuration
        plc1 = self.addHost('plc1', ip=IP['plc1'] + NETMASK)
        plc2 = self.addHost('plc2', ip=IP['plc2'] + NETMASK)
        attacker = self.addHost('attacker', ip=IP['attacker'] + NETMASK)
        
        # Connect hosts to switch
        self.addLink(s1, plc1)
        self.addLink(s1, plc2)
        self.addLink(s1, attacker)
        
        # ---------------- SUPERVISORY NETWORK --------------  #
        supervisoryIP = '192.168.2.254/24'
        s2 = self.addSwitch('s2')
        self.addLink(s2, router, intfName2='r0-eth2', params2={'ip': supervisoryIP})
        
        # Add SCADA and attacker2 hosts
        scada = self.addHost('scada', ip=IP['scada'] + NETMASK)
        attacker2 = self.addHost('attacker2', ip=IP['attacker2'] + NETMASK)
        
        # Connect hosts to supervisory switch
        self.addLink(s2, scada)
        self.addLink(s2, attacker2)

def configure_routes(net):
    """Configure routing after network startup"""
    print("Configuring inter-network routes...")
    
    # Configure routes for field network hosts to reach supervisory network
    net['plc1'].cmd('route add -net 192.168.2.0/24 gw 192.168.1.254')
    net['plc2'].cmd('route add -net 192.168.2.0/24 gw 192.168.1.254')
    net['attacker'].cmd('route add -net 192.168.2.0/24 gw 192.168.1.254')
    
    # Configure routes for supervisory network hosts to reach field network
    net['scada'].cmd('route add -net 192.168.1.0/24 gw 192.168.2.254')
    net['attacker2'].cmd('route add -net 192.168.1.0/24 gw 192.168.2.254')
    
    # Add default routes as backup
    net['scada'].cmd('route add default gw 192.168.2.254')
    
    print("Routes configured successfully!")

# Usage example:
if __name__ == '__main__':
    from mininet.net import Mininet
    from mininet.cli import CLI
    from mininet.link import TCLink
    from mininet.node import CPULimitedHost
    from mininet.log import setLogLevel
    
    setLogLevel('info')
    
    # Create network
    topo = ScadaTopo()
    net = Mininet(topo=topo, host=CPULimitedHost, link=TCLink)
    
    # Start network
    net.start()
    
    # Configure routes
    configure_routes(net)
    
    # Test connectivity
    print("\n=== Testing connectivity ===")
    print("PLC1 to SCADA:")
    net['plc1'].cmd('ping -c 3 192.168.2.30')
    print("SCADA to PLC1:")
    net['scada'].cmd('ping -c 3 192.168.1.10')
    
    # Start CLI
    CLI(net)
    
    # Stop network
    net.stop()
