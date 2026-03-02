# Flow Viewer using FlowNet Weights

This is largely inspired by the [```flow-code-python```](https://github.com/Johswald/flow-code-python) repository.

## Requirements

This runs on Python 3.11, and requires the following packages :

```
tensorflow
cv2
numpy
```



## Write a ```.flo``` file using your model weights

This supposes that you have a ```model.py``` file containing a ```FlowNet``` class where the architecture of your CNN is defined.

You can then write the ```.flo``` file between two frames using the following bash line

```
python write.py --frame1 'pathtoframe1' --frame1 'pathtoframe2' --weights_path 'pathtoweights' --model_path 'pathtoyourmodel.py'
```

## Visualize a ```.flo``` file

Generates the color code, using the Middleburry Colour Wheel: 
```
python computeColor.py --flowfile sortie.flo --write True
```

