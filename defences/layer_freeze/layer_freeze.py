import json

def freeze_layers(json_path, network):
    with open(json_path) as json_file:
        data = json.load(json_file)

    layer_list = list(data.keys())
    data.pop('fc', None)
    layers = []

    if (network == "googlenet"):
        return list(data['conv 1'].keys())
    elif (network == "resnet18"):
        return list(data['layer 1'].keys())
    elif (network == "resnext50"):
        return list(data['layer 1'].keys()) + list(data['layer 2'].keys())
    else:
        layer_dict = dict(sorted(data.items(), key=lambda item: item[1]))
        layer_dict = list(layer_dict.keys())
        layer_dict = layer_dict[-1:]
        for layer in layer_dict:
            layers.append(layer)

        return layers