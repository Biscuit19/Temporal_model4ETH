min_timestamp,max_timestamp = 1438923669,1547885786 # zsu数据集时间跨度
day_span = 60 * 60 * 24
day_size = 7
week_size = 8

# amount, block_timestamp, -1, timewindow, from_address, to_address, trans_hash
# 时间窗口跨度
time_window_span = day_span * day_size * week_size
# 时间窗口数量
time_window_num = int((max_timestamp - min_timestamp) / time_window_span)
print(f'0 - {time_window_num} time-windows totally')