import collections
import importlib
import config
from plotters.AmpPhaPlotter import Plotter # Amplitude and Phase plotter
decoder = importlib.import_module(f'decoders.{config.decoder}') # This is also an import

def string_is_int(s):
    '''
    Check if a string is an integer
    '''
    try:
        int(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    pcap_filename = input('Pcap file name: ')

    if '.pcap' not in pcap_filename:
        pcap_filename += '.pcap'
    pcap_filepath = '/'.join([config.pcap_fileroot, pcap_filename])

    try:
        samples = decoder.read_pcap(pcap_filepath)
    except FileNotFoundError:
        print(f'File {pcap_filepath} not found.')
        exit(-1)

    if config.plot_samples:
        plotter = Plotter(samples.bandwidth)

    while True:
        command = input('> ')

        if 'help' in command:
            print(config.help_str)
        
        elif 'exit' in command:
            break

        elif ('-' in command) and \
            string_is_int(command.split('-')[0]) and \
            string_is_int(command.split('-')[1]):

            start = int(command.split('-')[0])
            end = int(command.split('-')[1])

            macAddress = list()
            for index in range(start, end+1):
                macid = samples.get_mac(index).hex()
                macid = ':'.join([macid[i:i + 2] for i in range(0, len(macid), 2)])
                macAddress.append(macid)
                # if macid == "90:32:4b:e7:82:9f":
                if macid == "c4:e1:a1:b6:5c:8d":
                    if config.print_samples:
                        samples.print(index)
                    if config.plot_samples:
                        csi = samples.get_csi(
                            index,
                            config.remove_null_subcarriers,
                            config.remove_pilot_subcarriers
                        )
                        plotter.update(csi)
                        # time.sleep(config.plot_animation_delay_s)

            element_counts = collections.Counter(macAddress)
            for element, count in element_counts.items():
                print(f"Element {element}: {count} times")

        elif string_is_int(command):
            index = int(command)

            if config.print_samples:
                samples.print(index)
            if config.plot_samples:
                    csi = samples.get_csi(
                        index,
                        config.remove_null_subcarriers,
                        config.remove_pilot_subcarriers
                    )
                    plotter.update(csi)
        else:
            print('Unknown command. Type help.')