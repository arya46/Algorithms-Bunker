from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import tensorflow as tf

class Attention(Layer):
    
    """Class for Attention Layer"""
    
    def __init__(self, **kwargs):
        """
        Constructor Method
        """
        super(Attention, self).__init__(**kwargs)

    def call(self, query, key):
        """
        This method defines the main logic of Attention.
        
        Inputs:
            query: current time-step decoder hidden state  
                -> (batch_size, hidden_dim) 
            key: all encoder all hidden state           
                -> (batch_size, seq_len, hidden_dim) 
        
        Returns:
            Returns the context vector
                -> (batch_size, 1, hidden_dim)
                
        Note: The DOT scoring methods requires that the embedding dimension of
        both encoder and decoder are same for it to work.
        """

        # step 0: fixing the dimension of query
        query = tf.expand_dims(query, 1)                 # (batch_size, 1, hidden_dim)

        # step 1. calculate alignment score 
        scores = tf.matmul(query, key, transpose_b=True) # (batch_size, 1, seq_len)
        
        # step 2. apply softmax to the alignment scores  # (batch_size, 1, seq_len)
        a = tf.nn.softmax(scores)
        
        # step 3. multiply the softmaxed scores to "key" to get the "context vector"
        c_vector = tf.matmul(a, tf.cast(key, dtype=tf.float32)) 
        # (b, 1, seq_len) * (b, seq_len, hidden_dim)    -> (batch_size, 1, hidden_dim)

        return c_vector
    
    def get_config(self):
        """
        This method collects the input shape and other information about the model.
        """
        return super(Attention,self).get_config()