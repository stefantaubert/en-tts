import torch


def get_default_device():
  if torch.cuda.is_available():
    return torch.device("cuda:0")
  return torch.device("cpu")



def get_sample_count(sampling_rate: int, duration_s: float):
  return int(round(sampling_rate * duration_s, 0))

