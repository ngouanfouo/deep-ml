import numpy as np


class LSTM:

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def _sigmoid(self, v):
        return 1 / (1 + np.exp(-v))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """Processes a sequence of inputs and returns the hidden states,

        final hidden state, and final cell state.

        Args:
            x: Input sequence of shape (seq_len, input_size)
            initial_hidden_state: Initial hidden state of shape (hidden_size, 1)
            initial_cell_state: Initial cell state of shape (hidden_size, 1)
        """
        h = initial_hidden_state
        c = initial_cell_state
        hidden_states = []

        # Iterate through every time step in the input sequence
        for x_t in x:
            # Reshape x_t to a column vector: (input_size, 1)
            x_t = x_t.reshape(-1, 1)

            # Concatenate previous hidden state and current input: (input_size + hidden_size, 1)
            concat = np.vstack((h, x_t))

            # 1. Forget Gate
            f_t = self._sigmoid(np.dot(self.Wf, concat) + self.bf)

            # 2. Input Gate
            i_t = self._sigmoid(np.dot(self.Wi, concat) + self.bi)

            # 3. Candidate Cell State
            c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)

            # 4. Update Cell State
            c = f_t * c + i_t * c_tilde

            # 5. Output Gate & Hidden State Update
            o_t = self._sigmoid(np.dot(self.Wo, concat) + self.bo)
            h = o_t * np.tanh(c)

            # Store current hidden state squeezing out unnecessary dims if needed
            hidden_states.append(h.copy())

        return hidden_states, h, c