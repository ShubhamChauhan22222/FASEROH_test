import random
import numpy as np
import sympy as sp


class GenerateFunction:
    def __init__(self, vocab=None, max_depth=3):
        if vocab is None:
            self.vocab = np.array([
                ['add', 6, 2], ['sub', 3, 2], ['mul', 6, 2], ['div', 3, 2],
                ['pow', 3, 2], ['sq', 2, 1], ['sqrt', 2, 1], ['cb', 2, 1],
                ['cbrt', 2, 1], ['exp', 2, 1], ['ln', 2, 1], ['sin', 2, 1],
                ['cos', 2, 1], ['tan', 2, 1], ['asin', 2, 1], ['acos', 2, 1],
                ['atan', 2, 1], ['sinh', 2, 1], ['cosh', 2, 1], ['tanh', 2, 1],
                ['asinh', 2, 1], ['acosh', 2, 1], ['atanh', 2, 1], ['x', 10, 0]
            ])
        else:
            self.vocab = vocab
        self.max_depth = max_depth
        self.x = sp.symbols('x')

    def generate_expression(self, max_depth=None, depth=0, vocab=None):
        if vocab is None:
            vocab = self.vocab
        if max_depth is None:
            max_depth = self.max_depth

        if depth >= max_depth:
            return ['x']

        weights = vocab[:, 1].astype('float32')
        probs = weights / np.sum(weights)
        N = len(vocab)
        expr = []
        rand_idx = np.random.choice(N, p=probs)
        cur_token = vocab[rand_idx, 0]
        cur_arity = int(vocab[rand_idx, 2])
        expr.append(cur_token)

        if cur_arity == 0:
            return expr
        else:
            token_families = [
                ['sin', 'cos', 'tan', 'asin', 'acos', 'atan'],
                ['sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh'],
                ['exp', 'ln'], ['sq', 'sqrt'], ['cb', 'cbrt']
            ]
            token_family = next((f for f in token_families if cur_token in f), None)
            new_vocab = vocab
            if token_family:
                mask = np.isin(vocab[:, 0], token_family, invert=True)
                new_vocab = vocab[mask]

            if cur_arity == 1:
                child = self.generate_expression(max_depth, depth + 1, new_vocab)
                return expr + child
            elif cur_arity == 2:
                child1 = self.generate_expression(max_depth, depth + 1, new_vocab)
                child2 = self.generate_expression(max_depth, depth + 1, new_vocab)
                return expr + child1 + child2

    def sequence_to_sympy(self, expr):
        cur_token = expr[0]
        try:
            return float(cur_token)
        except ValueError:
            pass

        cur_idx = np.where(self.vocab[:, 0] == cur_token)[0][0]
        cur_arity = int(self.vocab[cur_idx, 2])

        if cur_arity == 0:
            return self.x
        elif cur_arity == 1:
            operand = self.sequence_to_sympy(expr[1:])
            return self._handle_unary(cur_token, operand)
        elif cur_arity == 2:
            left_tokens, right_tokens = self._split_expression(expr)
            left_expr = self.sequence_to_sympy(left_tokens)
            right_expr = self.sequence_to_sympy(right_tokens)
            return self._handle_binary(cur_token, left_expr, right_expr)

    def _handle_unary(self, operator, operand):
        operators = {
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'asinh': sp.asinh, 'acosh': sp.acosh, 'atanh': sp.atanh,
            'exp': sp.exp, 'ln': sp.log,
            'sq': lambda x: x ** 2, 'sqrt': sp.sqrt,
            'cb': lambda x: x ** 3, 'cbrt': sp.cbrt
        }
        return operators[operator](operand)

    def _handle_binary(self, operator, left, right):
        operators = {
            'add': lambda a, b: a + b, 'sub': lambda a, b: a - b,
            'mul': lambda a, b: a * b, 'div': lambda a, b: a / b,
            'pow': lambda a, b: a ** b
        }
        return operators[operator](left, right)

    def _split_expression(self, expr):
        arity_count = 1
        idx_split = 1
        for token in expr[1:]:
            try:
                float(token)
                arity_count -= 1
            except:
                idx = np.where(self.vocab[:, 0] == token)[0][0]
                arity_count += int(self.vocab[idx, 2]) - 1
            if arity_count == 0:
                break
            idx_split += 1
        return expr[1:idx_split], expr[idx_split:]

    def generate_functions(self, num_samples):
        functions = []
        while len(functions) < num_samples:
            try:
                expr = self.generate_expression()
                func = self.sequence_to_sympy(expr)
                functions.append(func)
            except:
                continue
        return functions[:num_samples]