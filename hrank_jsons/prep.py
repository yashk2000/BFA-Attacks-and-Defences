import json

# # resnet18
# layer_ranking = {
#     "Layer 1" : {
#         "Layer1.0.conv1" : 55.83799, 
#         "Layer1.0.conv2" : 55.865845,
#         "Layer1.1.conv1" : 55.910255,
#         "Layer1.1.conv2" : 55.97559
#     },
#     "Layer 2" : {
#         "Layer2.0.conv1" : 27.997513,
#         "Layer2.0.conv2" : 27.997683,
#         "Layer2.0.downsample.0" : 27.992077,
#         "Layer2.0.downsample.1" : 27.986876,
#         "Layer2.1.conv1" : 27.993711,
#         "Layer2.1.conv2" : 27.998768
#     }, 
#     "Layer 3" : {
#         "Layer3.0.conv1" : 13.996014,
#         "Layer3.0.conv2" : 13.99972,
#         "Layer3.0.downsample.0" : 13.999835,
#         "Layer3.0.downsample.1" : 13.999853,
#         "Layer3.1.conv1" : 13.999457,
#         "Layer3.1.conv2" : 13.999506
#     },
#     "Layer 4" : {
#         "Layer4.0.conv1" : 6.9999695,
#         "Layer4.0.conv2" : 6.9999485,
#         "Layer4.0.downsample.0" : 6.9999847,
#         "Layer4.0.downsample.1" : 6.9999638,
#         "Layer4.1.conv1" : 6.999869,
#         "Layer4.1.conv2" : 6.9998655
#     },
#     "FC Layer" : "linear"
# }

# # alexnet
# layer_ranking = {
#     "Features.0" : 54.805443,
#     "Features.3" : 26.987745,
#     "Features.6" : 12.999336,
#     "Features.8" : 12.999616,
#     "Features.10" : 12.99942,
#     "FC" : "linear"
# }

# # vgg19
# layer_ranking = {
#     "Features.0" : 31.538599,
#     "Features.2" : 31.974121,
#     "Features.5" : 15.999317,
#     "Features.7" : 15.9994755,
#     "Features.10" : 7.999939,
#     "Features.12" : 7.999927,
#     "Features.14" : 7.9999514,
#     "Features.16" : 7.999945,
#     "Features.19" : 3.9999938,
#     "Features.21" : 3.999997,
#     "Features.23" : 3.9999938,
#     "Features.25" : 3.999991,
#     "Features.28" : 2,
#     "Features.30" : 2,
#     "Features.32" : 2,
#     "Features.34" : 2,
#     "FC" : "classifier.6"
# }


# # resnext50
# layer_ranking = {
#     "Layer 1" : {
#         "conv1" : 107.78548
#     }, 
#     "Layer 2" : {
#         "Layer1.0.conv1": 55.84651,
#         "Layer1.0.conv2" : 55.929005,
#         "Layer1.1.conv1" : 55.941357,
#         "Layer1.1.conv2" : 55.94812,
#         "Layer1.2.conv1" : 55.92222,
#         "Layer1.2.conv2" : 55.903076
#     }, 
#     "Layer 3" : {
#         "Layer2.0.conv1" : 27.995216,
#         "Layer2.0.conv2" : 27.995092,
#         "Layer2.0.downsample.0" : 27.98473,
#         "Layer2.0.downsample.1" : 27.966919,
#         "Layer2.1.conv1" : 27.989819,
#         "Layer2.1.conv2" : 27.995667,
#         "Layer2.2.conv1" : 27.994213,
#         "Layer2.2.conv2" : 27.995802,
#         "Layer2.3.conv1" : 27.9959,
#         "Layer2.3.conv2" : 27.996471,
#     }, 
#     "Layer 4" : {
#         "Layer3.0.conv1" : 13.99939,
#         "Layer3.0.conv2" : 13.999391,
#         "Layer3.0.downsample.0" : 13.999414,
#         "Layer3.0.downsample.1" : 13.999334,
#         "Layer3.1.conv1" : 13.999409,
#         "Layer3.1.conv2" : 13.999451,
#         "Layer3.2.conv1" : 13.999414,
#         "Layer3.2.conv2" : 13.999487,
#         "Layer3.3.conv1" : 13.999336,
#         "Layer3.3.conv2" : 13.999462,
#         "Layer3.4.conv1" : 13.999329,
#         "Layer3.4.conv2" : 13.999347,
#         "Layer3.5.conv1" : 13.999256,
#         "Layer3.5.conv2" : 13.999237
#     },
#     "Layer 5" : {
#         "Layer4.0.conv1" : 6.999918,
#         "Layer4.0.conv2" : 6.999921,
#         "Layer4.0.downsample.0" : 6.9998474,
#         "Layer4.0.downsample.1" : 6.999921,
#         "Layer4.1.conv1" : 6.999921,
#         "Layer4.1.conv2" : 6.99993,
#         "Layer4.2.conv1" : 6.9999294,
#         "Layer4.2.conv2" : 6.9999084
#     },
#     "FC" : "fc"
# }

# # googlenet

# layer_ranking = {
#     "conv 1" : {
#         "conv1.conv" : 110.87039,
#         "conv2.conv" : 55.794434,
#         "conv3.conv" : 55.87343
#     },
#     "inception 3" : {
#         "inception3a.branch1.conv" : 27.956886,
#         "inception3a.branch2[0].conv" : 27.918427,
#         "inception3a.branch2[1].conv" : 27.975464,
#         "inception3a.branch3[0].conv" : 27.969238,
#         "inception3a.branch3[1].conv" : 27.98413,
#         "inception3a.branch4[1].conv" : 27.94839,
#         "inception3b.branch1.conv" : 27.987745,
#         "inception3b.branch2[0].conv" : 27.988602,
#         "inception3b.branch2[1].conv" : 27.984163,
#         "inception3b.branch3[0].conv" : 27.976954,
#         "inception3b.branch3[1].conv" : 27.996191,
#         "inception3b.branch4[1].conv" : 27.95012
#     },
#     "inception 4" : {
#         "inception4a.branch1.conv" : 13.999715,
#         "inception4a.branch2[0].conv" : 13.999756,
#         "inception4a.branch2[1].conv" : 13.999647,
#         "inception4a.branch3[0].conv" : 13.996387,
#         "inception4a.branch3[1].conv" : 13.999023,
#         "inception4a.branch4[1].conv" : 13.998462,
#         "inception4b.branch1.conv" : 13.999746,
#         "inception4b.branch2[0].conv" : 13.99968,
#         "inception4b.branch2[1].conv" : 13.989481,
#         "inception4b.branch3[0].conv" : 13.99974,
#         "inception4b.branch3[1].conv" : 13.99463,
#         "inception4b.branch4[1].conv" : 13.998901,
#         "inception4c.branch1.conv" : 13.998146,
#         "inception4c.branch2[0].conv" : 13.999683,
#         "inception4c.branch2[1].conv" : 13.999555,
#         "inception4c.branch3[0].conv" : 13.999741,
#         "inception4c.branch3[1].conv" : 13.999268,
#         "inception4c.branch4[1].conv" : 13.999097,
#         "inception4d.branch1.conv" : 13.999806,
#         "inception4d.branch2[0].conv" : 13.999706,
#         "inception4d.branch2[1].conv" : 13.999544,
#         "inception4d.branch3[0].conv" : 13.999756,
#         "inception4d.branch3[1].conv" : 13.999268,
#         "inception4d.branch4[1].conv" : 13.998974,
#         "inception4e.branch1.conv" : 13.999744,
#         "inception4e.branch2[0].conv" : 13.999786,
#         "inception4e.branch2[1].conv" : 13.999454,
#         "inception4e.branch3[0].conv" : 13.999609,
#         "inception4e.branch3[1].conv" : 13.999,
#         "inception4e.branch4[1].conv" : 13.999011
#     }, 
#     "inception 5" : {
#         "inception5a.branch1.conv" : 6.9999757,
#         "inception5a.branch2[0].conv" : 6.9999313,
#         "inception5a.branch2[1].conv" : 6.9999175,
#         "inception5a.branch3[0].conv" : 7,
#         "inception5a.branch3[1].conv" : 6.999817,
#         "inception5a.branch4[1].conv" : 6.9998045,
#         "inception5b.branch1.conv" : 6.9999633,
#         "inception5b.branch2[0].conv" : 6.9999185,
#         "inception5b.branch2[1].conv" : 6.9997153,
#         "inception5b.branch3[0].conv" : 6.999935,
#         "inception5b.branch3[1].conv" : 6.9965205,
#         "inception5b.branch4[1].conv" : 6.999939
#     }, 
#     "FC" : "fc"
# }

# # squeezenet

# layer_ranking  = {
#     "features.0" : 108.26249,
#     "features.3.squeeze" : 53.79844,
#     "features.4.squeeze" : 53.787403,
#     "features.5.squeeze" : 53.936623,
#     "features.6" : 26.441456,
#     "features.7.squeeze" : 26.982225,
#     "features.8.squeeze" : 26.992645,
#     "features.9.squeeze" : 26.9973,
#     "features.10.squeeze" : 26.995068,
#     "features.11" : 9.805716,
#     "classifier.1" : 12.999714
# }

# # shufflenet

# layer_ranking = {
#     "conv1.0" : 110.586784,
#     "stage2[0].branch1[2]" : 27.886848,
#     "stage2[0].branch2[0]" : 55.613342,
#     "stage2[1].branch2[0]" : 27.921682,
#     "stage2[3].branch2[0]" : 27.964258,
#     "stage3[0].branch2[0]" : 27.987597,
#     "stage3[2].branch2[5]" : 13.99987,
#     "stage4[0].branch2[0]" : 13.999691,
#     "stage4[2].branch2[5]" : 6.9999185,
#     "FC" : "fc"
# }

# mobilenet 

layer_ranking = {
    "features.0.0" : 78.714645,
    "features.1.conv.0.0" : 111.487686,
    "features.2.conv.0.0" : 109.394615,
    "features.2.conv.1.0" : 54.422947,
    "features.3.conv.0.0" : 54.852234,
    "features.3.conv.1.0" : 54.67948,
    "features.4.conv.0.0" : 55.24448,
    "features.4.conv.1.0" : 27.62284,
    "features.5.conv.0.0" : 27.348196,
    "features.5.conv.1.0" : 26.058447,
    "features.6.conv.0.0" : 27.205055,
    "features.6.conv.1.0" : 25.395662,
    "features.7.conv.0.0" : 27.166855,
    "features.7.conv.1.0" : 13.694848,
    "features.8.conv.0.0" : 13.327637,
    "features.8.conv.1.0" : 11.884089,
    "features.9.conv.0.0" : 13.1082115,
    "features.9.conv.1.0" : 11.803967,
    "features.10.conv.0.0" : 12.830086,
    "features.10.conv.1.0" : 11.389217,
    "features.11.conv.0.0" : 13.451523,
    "features.11.conv.1.0" : 12.945712,
    "features.12.conv.0.0" : 12.492089,
    "features.12.conv.1.0" : 11.142885,
    "features.13.conv.0.0" : 12.396853,
    "features.13.conv.1.0" : 10.96796,
    "features.14.conv.0.0" : 11.148429,
    "features.14.conv.1.0" : 6.373972,
    "features.15.conv.0.0" : 6.088983,
    "features.15.conv.1.0" : 5.010454,
    "features.16.conv.0.0" : 5.682835,
    "features.16.conv.1.0" : 4.8759665,
    "features.17.conv.0.0" : 3.5290008,
    "features.17.conv.1.0" : 5.1528306,
    "features.18.0" : 4.6817846,
    "FC" : "classifier.1"
}

layer_ranking = eval(repr(layer_ranking).lower())

with open('mobilenet.json', 'w') as fp:
    json.dump(layer_ranking, fp)