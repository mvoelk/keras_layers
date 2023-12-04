import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model


def plot_feature_statistic(models, x):
    if type(models) not in [list, tuple]:
        models = [models]
    plt.figure(figsize=(16,4))
    for i, model in enumerate(models):
        m = Model(model.input, [l.output for l in model.layers])
        y = m(x)
        mean = [np.mean(a) for a in y]
        std = [np.std(a) for a in y]
        var = [np.var(a) for a in y]
        names = ['%s\n%s'%(l.name,l.output_shape[1:]) for l in model.layers]
        n = np.arange(len(names)) + i*0.1
        plt.errorbar(n, mean, yerr=std, 
                     marker='x', linestyle='None', capsize=5, elinewidth=1, 
                     markeredgewidth=1, markersize=8, label=model.name)
    if len(models) == 1:
        plt.xticks(n, names, rotation=25)
    plt.hlines(0, [-0.2], [len(n)-1+len(models)*0.1+0.2], 'k', linestyles='--', linewidth=1, alpha=0.5)
    plt.grid(); plt.legend(); plt.title('activation mean and std')
    plt.show()


def plot_feature_statistic_with_mask(models, xm):
    if type(models) not in [list, tuple]:
        models = [models]
    
    plt.figure(figsize=(16,4))
    for i, model in enumerate(models):
        conv_layers = [l for l in model.layers if l.__class__.__name__.find('Conv2D') != -1]
        outputs = [model.inputs] + [l.output for l in conv_layers]
        
        xms = Model(model.input, outputs)(xm)
        
        features, masks = zip(*xms)
        weighted_features = [x*m for x, m in xms]
        y = features
        y = weighted_features
        #y = masks
        
        mean = [np.mean(a) for a in y]
        std = [np.std(a) for a in y]
        var = [np.var(a) for a in y]
        layer_names = ['input'] + [l.name for l in conv_layers]
        names = ['%s\n%s'%(n, o[0].shape[1:]) for n, o in zip(layer_names, outputs)]
        n = np.arange(len(names)) + i*0.1
        plt.errorbar(n, mean, yerr=std, 
                     marker='x', linestyle='None', capsize=5, elinewidth=1, 
                     markeredgewidth=1, markersize=8, label=model.name)
    if len(models) == 1:
        plt.xticks(n, names, rotation=25)
    plt.hlines(0, [-0.2], [len(n)-1+len(models)*0.1+0.2], 'k', linestyles='--', linewidth=1, alpha=0.5)
    plt.grid(); plt.legend(); plt.title('activation mean and std')
    plt.show()


def plot_feature_activation(models, x, same_vminmax=True):
    """
    # Arguments
        models: list of keras models with layes
            Input
            Conv2D or similar
            Conv2D or similar
            Conv2D or similar
            ...
        x: input data
    
    # Plot of layer output
        features
        features * mask
        mask
    """

    if type(models) not in [list, tuple]:
        models = [models]
    
    def calc_min_max(features):
        # TODO: mean and std?
        vmin_each, vmax_each = [np.min(a) for a in features], [np.max(a) for a in features]
        vmin, vmax = np.min(vmin_each), np.max(vmax_each)
        return vmin, vmax, vmin_each, vmax_each
    
    for i, model in enumerate(models):
        m = Model(model.input, [l.output for l in model.layers])
        y = m(x)
        vmin, vmax, vmin_each, vmax_each = calc_min_max(y)
        print('%-20s  %10.3f %10.3f '% (model.name, vmin, vmax))
        plt.figure(figsize=(17, 2))
        for j in range(min(len(y), 8)):
            plt.subplot(181+j)
            if same_vminmax:
                plt.imshow(y[j][0,...,0], vmin=vmin, vmax=vmax)
            else:
                plt.imshow(y[j][0,...,0], vmin=vmin_each[j], vmax=vmax_each[j])
            plt.title('%.3f %.3f'%(vmin_each[j], vmax_each[j]))
        plt.show()


def plot_feature_activation_with_masks(models, xm, same_vminmax=True):
    """
    # Arguments
        models: list of keras models with layes
            Input for features
            Input for mask
            PartialConv2D or similar
            PartialConv2D or similar
            PartialConv2D or similar
            ...
        xm: input data, list [features, mask]
    
    # Plot of layer output
        features
        features * mask
        mask
    """
    
    if type(models) not in [list, tuple]:
        models = [models]
    
    def calc_min_max(features):
        vmin_each, vmax_each = [np.min(a) for a in features], [np.max(a) for a in features]
        vmin, vmax = np.min(vmin_each), np.max(vmax_each)
        return vmin, vmax, vmin_each, vmax_each
    
    for i, model in enumerate(models):
        conv_layers = [l for l in model.layers if l.__class__.__name__.find('Conv2D') != -1]
        outputs = [model.inputs] + [l.output for l in conv_layers]
        
        xms = Model(model.input, outputs)(xm)
        
        features, masks = zip(*xms)
        weighted_features = [x*m for x, m in xms]
        
        min_max_x = calc_min_max(features)
        min_max_m = calc_min_max(masks)
        min_max_xm = calc_min_max(weighted_features)
        
        print('%-20s  x: %10.3f %10.3f  xm: %10.3f %10.3f  m: %10.3f %10.3f' % 
              (model.name, min_max_x[0], min_max_x[1], min_max_xm[0], min_max_xm[1], min_max_m[0], min_max_m[1]))
        
        plt.figure(figsize=(17, 2))
        for j in range(min(len(features), 8)):
            plt.subplot(181+j); plt.imshow(features[j][0,...,0], vmin=min_max_x[0], vmax=min_max_x[1])
            plt.title('%.3f %.3f'%(min_max_x[2][j], min_max_x[3][j]))
        plt.show()
        
        plt.figure(figsize=(17, 2))
        for j in range(min(len(masks), 8)):
            plt.subplot(181+j)
            if same_vminmax:
                plt.imshow(weighted_features[j][0,...,0], vmin=min_max_xm[0], vmax=min_max_xm[1])
            else:
                plt.imshow(weighted_features[j][0,...,0], vmin=min_max_xm[2][j], vmax=min_max_xm[3][j])
            plt.title('%.3f %.3f'%(min_max_xm[2][j], min_max_xm[3][j]))
        plt.show()
        
        plt.figure(figsize=(17, 2))
        for j in range(min(len(masks), 8)):
            plt.subplot(181+j)
            plt.imshow(masks[j][0,...,0], vmin=0, vmax=1, cmap='gray')
            plt.title('%.3f %.3f'%(min_max_m[2][j], min_max_m[3][j]))
        plt.show()


