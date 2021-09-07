from ct1_package import load_data, filter_device, load_folder, resample, extract_file_features, split_data, ten_fold
import matplotlib.pyplot as plt


''' Input Parameter '''
step_size = '100ms'
window_size = 100
window_overlap = 0.5
folder_path = r"D:\Uni\dataset"
converted_files = False
use_features = {'mean': True,
                'std': True,
                'var': False,
                'min': True,
                'max': True}

load_sensors = {'phone_gyro': True,
                'phone_accel': False,
                'watch_gyro': True,
                'watch_accel': True}

# file = load_data(os.path.join(folder_path, 'p1_phone_watch_sensorFusion.ctm'))
# filter_device(group_label(file)['hand_wash'][0], 'phone', 'GYRO').plot()
# filter_device(group_label(file)['hand_wash'][0], 'phone', 'ACC').plot()
# filter_device(group_label(file)['sitting'][0], 'phone', 'GYRO').plot()
# filter_device(group_label(file)['downstairs'][0], 'phone', 'ACC').plot()
# filter_device(group_label(file)['hand_wash'][0], 'phone', 'ACC').plot()
# filter_device(group_label(file)['sitting'][0], 'phone', 'ACC').plot()
# plt.show()

''' Loading Data'''
files = load_folder(folder_path, sensors=load_sensors, converted=converted_files)

''' Resampling Data '''
print("Resampling: ", end='')
files = [resample(file, step_size) for file in files]
print()

''' Extracting Features'''
x_files, y_files = extract_file_features(files, window_size, window_overlap, use_features)

''' Split Data for 10-fold'''
training_data, validation_data = split_data(x_files, y_files)

''' Ten Fold Validation'''
ten_fold(training_data, validation_data)