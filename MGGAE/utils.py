# based on the Github code of Van den Berg https://github.com/riannevdberg/gc-mc
#modified by Juliette Valenchon

from __future__ import division
from __future__ import print_function


def construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero, v_features_nonzero,
                        support, support_t, labels, indices_labels, u_indices, v_indices,
                        dropout):
    """
    Function that creates feed dictionary when running tensorflow sessions.
    Inputs:
        placeholders: dictionary that defines the variables and that will be filled with the variables defined  in the function
        u_features, v_features, u_features_nonzero, v_features_nonzero, support, support_t, labels, indices_labels, u_indices, v_indices, dropout: variables
        to fill the dictionary
    Outputs: Feed dictionary for tensorflow algorithm
    """

    feed_dict = dict()
    feed_dict.update({placeholders['u_features']: u_features})
    feed_dict.update({placeholders['v_features']: v_features})
    feed_dict.update({placeholders['u_features_nonzero']: u_features_nonzero})
    feed_dict.update({placeholders['v_features_nonzero']: v_features_nonzero})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['support_t']: support_t})

    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['indices_labels']: indices_labels})

    feed_dict.update({placeholders['user_indices']: u_indices})
    feed_dict.update({placeholders['item_indices']: v_indices})

    feed_dict.update({placeholders['dropout']: dropout})

    return feed_dict
