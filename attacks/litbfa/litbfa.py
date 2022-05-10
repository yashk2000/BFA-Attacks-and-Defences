import json

def attack_layers(json_path, network):
    with open(json_path) as json_file:
        data = json.load(json_file)

    depth = 3

    layer_list = list(data.keys())

    layers = [data['fc']]
    depth -= 1

    if (network == "googlenet" or network == "resnet18" or network == "resnext50"):
        for i in range(depth):
            layer_dict = data[layer_list[depth - i]]
            layer_dict = list(layer_dict.keys())
            for layer in layer_dict:
                layers.append(layer)

        return layers
    else:
        data.pop('fc', None)
        layer_dict = dict(sorted(data.items(), key=lambda item: item[1]))
        layer_dict = list(layer_dict.keys())
        layer_dict = layer_dict[-depth:]
        for layer in layer_dict:
            layers.append(layer)

        return layers
        