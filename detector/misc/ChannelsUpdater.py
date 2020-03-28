from time import sleep

from obspy import UTCDateTime


class ChannelsUpdater:

    def __init__(self):
        self.dic = {}

    def update(self, station, channels):
        if station not in self.dic:
            self.dic[station] = {}
        self.dic[station].update({ch: UTCDateTime() for ch in channels})

    def get_channels_dic(self):
        ret_val = {}
        for station in self.dic:
            chs_dic = self.dic[station]
            chs = [ch.decode() for ch in chs_dic if chs_dic[ch] > UTCDateTime() - 5]
            if chs:
                ret_val[station.decode()] = sorted(chs)
        return ret_val

# channelsUpdater = ChannelsUpdater()
# print(str(channelsUpdater.get_channels_dic()))
# channelsUpdater.update(b'nd01', [b'x', b'y', b'z'])
# print(str(channelsUpdater.get_channels_dic()))
# sleep(3)
# channelsUpdater.update(b'nd01', [b'x'])
# sleep(3)
# print(str(channelsUpdater.get_channels_dic()))

