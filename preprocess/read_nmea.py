import pynmea2
import numpy as np
from helpers import convert_time_to_rostime,devide_name_to_time

'''
dvs_vpr_2020-04-24-15-12-03
dvs_vpr_2020-04-27-18-13-29
dvs_vpr_2020-04-28-09-14-11
dvs_vpr_2020-04-29-06-20-23
'''

def get_gps(nmea_file_path):
    nmea_file = open(nmea_file_path, encoding='utf-8')

    latitudes, longitudes, timestamps = [], [], []

    first_timestamp = None
    previous_lat, previous_lon = 0, 0
    output_file = open('nmea_time.txt', 'w')
    for line in nmea_file.readlines():
        try:
            msg = pynmea2.parse(line)
            if first_timestamp is None:
                first_timestamp = msg.timestamp
            if msg.sentence_type not in ['GSV', 'VTG', 'GSA']:
                output_file.write("time:"+f"hour{msg.timestamp.hour} minute{msg.timestamp.minute} second{msg.timestamp.second}"+'\n')
                #print(msg.timestamp, msg.latitude, msg.longitude)
                #print(repr(msg.latitude))
                dist_to_prev = np.linalg.norm(np.array([msg.latitude, msg.longitude]) - np.array([previous_lat, previous_lon]))
                if msg.latitude != 0 and msg.longitude != 0 and msg.latitude != previous_lat and msg.longitude != previous_lon and dist_to_prev > 0.0001:
                    timestamp_diff = (msg.timestamp.hour - first_timestamp.hour) * 3600 + (msg.timestamp.minute - first_timestamp.minute) * 60 + (msg.timestamp.second - first_timestamp.second)
                    latitudes.append(msg.latitude); longitudes.append(msg.longitude); timestamps.append(timestamp_diff)
                    previous_lat, previous_lon = msg.latitude, msg.longitude

        except pynmea2.ParseError as e:
            # print('Parse error: {} {}'.format(msg.sentence_type, e))
            continue

    return np.array(np.vstack((latitudes, longitudes, timestamps))).T

'''
# 示例：计算两点间的距离
lat1, lon1 = 40.7128, -74.0060  # 纽约
lat2, lon2 = 40.7157, -74.0030  # 纽约附近
distance = calculate_distance(lat1, lon1, lat2, lon2)
print(f"Distance: {distance:.2f} meters")
'''
def process_gps(file_name):
    file_path = f'/root/autodl-fs/Brizbane_dataset/{file_name}.nmea'
    year,month,day,hour,minute,second = devide_name_to_time(file_name)
    gps_data = get_gps(file_path)
    gps_data[:, 2] = gps_data[:, 2].astype(float) + convert_time_to_rostime(year,month,day,hour,minute,second,0) #sr
     # 将gps_data写入txt文件
    with open(f'/root/autodl-tmp/processed_data/gps_output/{file_name}.txt', 'w') as f:
        for row in gps_data:
            f.write(' '.join(map(str, row)) + '\n')
    return gps_data

#process_gps('20200421_170039-sunset1_concat')
#process_gps('20200422_172431-sunset2_concat')
process_gps('20200424_151015-daytime_concat')
#process_gps('20200427_181204-night_concat')
#process_gps('20200429_061912-sunrise_concat')
#process_gps('20200429_061912-sunrise_concat')
