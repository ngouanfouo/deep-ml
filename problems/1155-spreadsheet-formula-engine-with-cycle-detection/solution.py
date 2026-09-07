import re


class CycleError(Exception):
    """Raised when a circular reference is detected during cell evaluation."""
    pass


class FormulaParser:
    """Recursive descent parser for spreadsheet formulas."""
    def __init__(self, tokens, eval_cell_func):
        self.tokens = tokens
        self.pos = 0
        self.eval_cell = eval_cell_func

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def parse_expression(self):
        val = self.parse_term()
        while self.peek() in ('+', '-'):
            op = self.consume()
            right = self.parse_term()
            if op == '+':
                val += right
            else:
                val -= right
        return val

    def parse_term(self):
        val = self.parse_factor()
        while self.peek() == '*':
            self.consume()
            right = self.parse_factor()
            val *= right
        return val

    def parse_factor(self):
        tok = self.consume()
        if tok is None:
            raise ValueError("Unexpected end of expression")
        if tok.isdigit():
            return int(tok)
        else:
            return self.eval_cell(tok)


def evaluate(operations):
    """
    Processes a list of spreadsheet operations ("set" and "get") and returns
    the results of all "get" operations in order.
    """
    cells = {}
    outputs = []

    def eval_cell(cell, visiting):
        if cell in visiting:
            raise CycleError()
        if cell not in cells:
            return 0

        val = cells[cell]
        
        # If it's a numeric literal string
        if val.isdigit():
            return int(val)

        # If it's a formula starting with '='
        if val.startswith('='):
            formula_str = val[1:]
            token_pattern = re.compile(r'([A-Z]+\d+|\d+|[+\-*])')
            tokens = token_pattern.findall(formula_str)

            visiting.add(cell)
            try:
                parser = FormulaParser(tokens, lambda c: eval_cell(c, visiting))
                result = parser.parse_expression()
            finally:
                visiting.remove(cell)
            return result
        else:
            return int(val)

    for op in operations:
        cmd = op[0]
        if cmd == "set":
            _, cell, value = op
            cells[cell] = value
        elif cmd == "get":
            _, cell = op
            visiting = set()
            try:
                res = eval_cell(cell, visiting)
                outputs.append(res)
            except CycleError:
                outputs.append("CYCLE")

    return outputs