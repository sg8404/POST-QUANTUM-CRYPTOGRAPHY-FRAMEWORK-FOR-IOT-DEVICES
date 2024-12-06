
import ns.core
import ns.network
import ns.internet
import ns.point_to_point
import ns.applications

# Set up simulation parameters
num_nodes = 1000  # Change this for different network sizes
simulation_time = 60.0  # seconds

# Create nodes and configure network
nodes = ns.network.NodeContainer()
nodes.Create(num_nodes)

p2p = ns.point_to_point.PointToPointHelper()
p2p.SetDeviceAttribute("DataRate", ns.core.StringValue("10Mbps"))
p2p.SetChannelAttribute("Delay", ns.core.StringValue("2ms"))

devices = p2p.Install(nodes)

# Install internet stack
stack = ns.internet.InternetStackHelper()
stack.Install(nodes)

address = ns.internet.Ipv4AddressHelper()
address.SetBase(ns.network.Ipv4Address("10.1.1.0"), ns.network.Ipv4Mask("255.255.255.0"))
interfaces = address.Assign(devices)

# Set up traffic generator
on_off = ns.applications.OnOffHelper("ns3::UdpSocketFactory", ns.network.AddressValue(interfaces.GetAddress(1)))
on_off.SetAttribute("DataRate", ns.core.StringValue("1Mbps"))
on_off.SetAttribute("PacketSize", ns.core.UintegerValue(512))

app = on_off.Install(nodes.Get(0))
app.Start(ns.core.Seconds(1.0))
app.Stop(ns.core.Seconds(simulation_time))

# Enable packet tracing
ns.network.AsciiTraceHelper().EnablePcap("throughput_latency", devices)

# Run simulation
ns.core.Simulator.Stop(ns.core.Seconds(simulation_time))
ns.core.Simulator.Run()
ns.core.Simulator.Destroy()

print("Simulation complete. Throughput and latency data saved.")
