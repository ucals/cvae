import argparse
import pandas as pd
from sys import platform
if platform == 'linux':
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
from mnist import get_val_images
import cv2
from PIL import Image


fig = plt.figure(figsize=(8, 7))
spec = gridspec.GridSpec(nrows=2, ncols=3, hspace=.3, wspace=.25)
axl = fig.add_subplot(spec[0, :])
axd = [fig.add_subplot(spec[1, i]) for i in range(3)]
plt.subplots_adjust(left=0.10, bottom=0.05, right=0.90, top=0.90,
                    wspace=0.3, hspace=0.25)
im = Image.open('../data/pyro_logo.png')
im.thumbnail((900, 900), Image.ANTIALIAS)

im2 = im.copy()
im2.putalpha(30)
im.paste(im2, im)

im = np.array(im).astype(np.float) / 255
fig.figimage(im, 600, 850)  # 1200, 800


def animate(i, dfs, dfl, inputs, digits):
    if i < len(dfl):
        axl.clear()
        axl.set_ylim(top=200, bottom=60)
        axl.plot(dfl.iloc[0:i, 0].values, dfl.iloc[0:i, 1].values)
        axl.set_ylabel('Loss')
        axl.set_xlabel('Epochs')
        axl.set_title('Training Progress')

        s = f'Loss: {dfl.iloc[i - 10:i, 1].mean():.2f}'
        axl.text(0.85, 0.96, s, horizontalalignment='left',
                 verticalalignment='top', transform=axl.transAxes)
        s = f'Epoch: {dfl.iloc[i, 0]:.1f}'
        axl.text(0.85, 0.88, s, horizontalalignment='left',
                 verticalalignment='top', transform=axl.transAxes)

        data = dfs[dfs['epoch'] == dfl['epoch'].iloc[i]]
        if len(data) > 0:
            for j, k in enumerate([0, 18, 4]):  # index 15 also good 3-5 confusion
                img = data.iloc[k, :784].values.reshape(28, 28)
                inp = inputs[k]
                img[inp != -1] = inp[inp != -1]

                img = cv2.resize(img, dsize=(280, 280),
                                 interpolation=cv2.INTER_NEAREST)
                img = np.stack((img,)*3, axis=-1)
                img[140:, 140, 1] = 1
                img[140, :140, 1] = 1

                axd[j].clear()
                axd[j].imshow(img, cmap='gray')
                axd[j].set_title('Sample %d' % (digits[k]))
                axd[j].get_xaxis().set_visible(False)
                axd[j].get_yaxis().set_visible(False)


def main(args):
    dfs = pd.read_csv('../data/samples.csv')
    dfl = pd.read_csv('../data/losses.csv')

    inputs, digits = get_val_images(num_quadrant_inputs=1,
                                    num_images=30, shuffle=False)
    inputs = inputs.numpy()

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Carlos Souza'), bitrate=1800)

    ani = animation.FuncAnimation(fig, animate, interval=50, frames=100000,
                                  fargs=(dfs, dfl, inputs, digits, ))

    if args.show:
        plt.show()
    else:
        ani.save('animation.mp4', writer=writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate animation.')
    parser.add_argument('-s', '--show', action="store_true", default=False,
                        help='Use this flag to show video animation on screen'
                             'instead of saving it to file. Default is to save'
                             'to "animation.mp4".')
    args = parser.parse_args()
    main(args)












