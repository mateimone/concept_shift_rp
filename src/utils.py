def fed_avg(local_weights, dataset_size_per_client):
    avg_dict = {}
    sum_dataset = sum(dataset_size_per_client)
    for i, dictionary in enumerate(local_weights):
        for key, tensor in dictionary.items():
            if key not in avg_dict:
                avg_dict[key] = tensor.clone() * (dataset_size_per_client[i]/sum_dataset)
            else:
                avg_dict[key] += tensor.clone() * (dataset_size_per_client[i]/sum_dataset)
    return avg_dict
