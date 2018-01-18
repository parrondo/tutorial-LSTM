from pandas import read_csv
from matplotlib import pyplot

def visualize(interim_data):
    '''
    visualize interim data
    args:
         interim_data (str) interim data file full path name
    '''
    # load dataset
    dataset = read_csv(interim_data, header=0, index_col=0)
    values = dataset.values
    # specify columns to plot
    groups = [0, 1, 2, 3, 5, 6, 7]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()
