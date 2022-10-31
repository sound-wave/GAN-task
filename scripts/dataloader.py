import torch
import torchaudio


torchaudio.datasets.YESNO(
     root='./',
     url='http://www.openslr.org/resources/1/waves_yesno.tar.gz',
     folder_in_archive='waves_yesno',
     download=True)


yesno_data = torchaudio.datasets.YESNO('./', download=True)

n = 3
waveform, sample_rate, labels = yesno_data[n]
print("Waveform: {}\nSample rate: {}\nLabels: {}".format(waveform, sample_rate, labels))

data_loader = torch.utils.data.DataLoader(yesno_data,
                                          batch_size=1,
                                          shuffle=True)

for data in data_loader:
  print("Data: ", data)
  print("Waveform: {}\nSample rate: {}\nLabels: {}".format(data[0], data[1], data[2]))
  break