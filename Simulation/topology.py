#/usr/bin/python

import yaml
from mininet.cli import CLI
from mininet.log import lg, info
from mininet.topolib import TreeNet

# Load configuration from YAML
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)

if __name__ == '__main__':
    lg.setLogLevel('info')
    net = TreeNet(depth=1, fanout=5)

    # Assign IPs from YAML
    hosts_cfg = cfg['mininet']['hosts']
    for host in net.hosts:
        if host.name in hosts_cfg:
            host_ip = hosts_cfg[host.name]['ip']
            host.setIP(host_ip)

    # Add NAT connectivity
    net.addNAT().configDefault()
    
    net.start()

    # Print host info from YAML
    print("*** Mininet network started")
    print("*** Hosts:")
    for h, info_dict in hosts_cfg.items():
        print(f"   {h} ({info_dict['desc']}): {info_dict['ip']}")

    # Python path from YAML
    python_path = cfg.get('venv', {}).get('python_path', 'python3')

    # Example usage
    print("\n*** Example usage:")
    print(f"   mininet> h1 {python_path} scada_server.py > logs/scada.log 2>&1 &")
    print(f"   mininet> h5 {python_path} visualization_server.py > logs/visualization.log 2>&1 &")
    print(f"   mininet> h2 {python_path} real-time_controller.py > logs/controller.log 2>&1 &")
    print("\nType `exit` or press Ctrl-D to stop the network.")
    
    CLI(net)
    net.stop()

