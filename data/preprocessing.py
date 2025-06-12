import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_data(dataset, train=True):
    dims = dataset.shape
    if len(dims) == 3:
        dataset = dataset.values.reshape(-1, dims[1]*dims[2])
        if train:
            global train_scaler
            train_scaler = MinMaxScaler(feature_range=(-1, 1))
            normalized_data = train_scaler.fit_transform(dataset)
        else:
            normalized_data = train_scaler.transform(dataset)
    else:
        normalized_data = []
        for i in dataset.features:
            ds = dataset.sel({'features': i}).values.reshape(-1, dims[1]*dims[2])
            scaler = MinMaxScaler(feature_range=(-1, 1))
            if train:
                normalized_data.append(scaler.fit_transform(ds))
            else:
                normalized_data.append(scaler.transform(ds))
        normalized_data = np.stack(normalized_data, axis=-1)

    normalized_data = normalized_data.reshape(dims + (1,))
    return normalized_data

def load_and_prepare_data(coarse_data, fine_data, batch_size=8, shuffle=True):
    coarse_norm = normalize_data(coarse_data, train=True)
    fine_norm = normalize_data(fine_data, train=True)

    coarse_norm = coarse_norm.reshape(coarse_data.shape)
    fine_norm = fine_norm.reshape(fine_data.shape + (1,))

    if shuffle:
        indices = np.arange(coarse_norm.shape[0])
        np.random.shuffle(indices)
        coarse_norm = coarse_norm[indices]
        fine_norm = fine_norm[indices]

    def batch_generator(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

    coarse_batches = list(batch_generator(coarse_norm, batch_size))
    fine_batches = list(batch_generator(fine_norm, batch_size))

    return coarse_batches, fine_batches
