"""A safe math calculator tool.

Uses Python's AST to evaluate expressions without exposing eval() to arbitrary
code. Supports + - * / // % ** and parentheses, plus a whitelist of math
functions (sqrt, sin, log, factorial, ...) and constants (pi, e, tau).
Anything outside the whitelist is rejected.
"""

import ast
import math
import operator
from typing import Any, Dict

from .base import Tool

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Whitelisted callables the expression may use.
_FUNCS = {
    "sqrt": math.sqrt,
    "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "log": math.log,        # log(x) natural, log(x, base) for any base
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "degrees": math.degrees,
    "radians": math.radians,
    "hypot": math.hypot,
    "min": min,
    "max": max,
}

# Whitelisted constants.
_CONSTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numbers are allowed")
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError(f"unknown name: {node.id}")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("unsupported function")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        args = [_eval(a) for a in node.args]
        return _FUNCS[node.func.id](*args)
    raise ValueError("unsupported expression")


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate a math expression. Supports + - * / // % ** (), and functions "
        "like sqrt, sin, cos, tan, log, log10, exp, factorial, abs, round, and "
        "constants pi, e, tau. Example: 'sqrt(2) * sin(pi/4)'."
    )
    parameters = {"expression": "A math expression, e.g. 'sqrt(144) + log(100, 10)'."}

    def run(self, args: Dict[str, Any]) -> str:
        expr = str(args.get("expression", "")).strip()
        if not expr:
            return "error: no expression provided"
        try:
            tree = ast.parse(expr, mode="eval")
            result = _eval(tree)
            return str(result)
        except Exception as exc:
            return f"error: {exc}"
