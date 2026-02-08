from __future__ import annotations

import ast
import math
import operator
from typing import Any

from eval_infra.tools.base import Tool

# Safe operators for the calculator
_SAFE_OPS = {
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

_SAFE_FUNCS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
    "factorial": math.factorial,
    "gcd": math.gcd,
}


class Calculator(Tool):
    name = "calculator"
    description = "Evaluate a mathematical expression. Supports basic arithmetic, powers, and common math functions (sqrt, log, sin, cos, etc.)."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "The mathematical expression to evaluate, e.g. '2**10 + sqrt(144)'"}
        },
        "required": ["expression"],
    }

    def execute(self, expression: str = "", **kwargs: Any) -> str:
        try:
            result = self._safe_eval(expression)
            return str(result)
        except Exception as e:
            return f"Error evaluating expression: {e}"

    def _safe_eval(self, expr: str) -> Any:
        tree = ast.parse(expr, mode="eval")
        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, complex)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value!r}")
        elif isinstance(node, ast.BinOp):
            op_func = _SAFE_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op_func(self._eval_node(node.left), self._eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            op_func = _SAFE_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op_func(self._eval_node(node.operand))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
                func = _SAFE_FUNCS[node.func.id]
                args = [self._eval_node(arg) for arg in node.args]
                return func(*args)
            raise ValueError(f"Unsupported function call")
        elif isinstance(node, ast.Name):
            if node.id in _SAFE_FUNCS:
                val = _SAFE_FUNCS[node.id]
                if not callable(val):
                    return val
            raise ValueError(f"Unsupported name: {node.id}")
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")
