"""ONNX graph inspection, rewriting and budgeting.

Everything here is board-free and cheap. The design bet of the whole zoo is
that most models can be rejected — with a specific, quotable reason — before
any expensive stage runs.
"""
