# Copyright 2020 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Handy functions that are not necessary for the conceptual understanding of
# this code example

from collections import defaultdict

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import random

def get_groupings(sample):
    """Grab selected items and group them by color"""
    colored_points = defaultdict(list)

    for label, bool_val in sample.items():
        # Skip over items that were not selected
        if not bool_val:
            continue

        # Parse selected items
        # Note: label look like "<x_coord>,<y_coord>_<color>"
        coord, color = label.split("_")
        coord_tuple = tuple(map(float, coord.split(",")))
        colored_points[color].append(coord_tuple)

    return dict(colored_points)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def visualize_groupings(groupings_dict, filename):
    """
    Args:
        groupings_dict: key is a color, value is a list of x-y coordinate tuples.
          For example, {'r': [(0,1), (2,3)], 'b': [(8,3)]}
        filename: name of the file to save plot in
    """
    colors = []

    for val, points in groupings_dict.items():
        # Ignore items that do not contain any coordinates
        if not points:
            continue


        #color = str(get_cmap(count))
        shade = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) 
        if shade not in colors:
            colors.append(shade)
            color = shade

        print(color)

        # Populate plot
        #point_style = color + "o"
        plt.scatter(*zip(*points), c = color)

        """
        if val in centers_dict.keys():
            point_style = "v"
            plt.plot(centers_dict[val][0],centers_dict[val][1], c = color, point_style)

            """
        

    plt.savefig(filename)

def visualize_groupings_again(groupings_dict, filename):
    """
    Args:
        groupings_dict: key is a color, value is a list of x-y coordinate tuples.
          For example, {'r': [(0,1), (2,3)], 'b': [(8,3)]}
        filename: name of the file to save plot in
    """
        
    for color, points in groupings_dict.items():
        # Ignore items that do not contain any coordinates
        if not points:
            continue

        # Populate plot
        point_style = color + "o"
        plt.plot(*zip(*points), point_style)
        

    plt.savefig(filename)

def visualize_scatterplot(x_y_tuples_list, filename):
    """Plotting out a list of x-y tuples

    Args:
        x_y_tuples_list: A list of x-y coordinate values. e.g. [(1,4), (3, 2)]
    """
    plt.plot(*zip(*x_y_tuples_list), "o")
    plt.savefig(filename)

