#!/usr/bin/env python3

"""
Based upon Shor routine from qiskit 0.44
"""

import numpy as np
from typing import Union

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Instruction, Gate, ParameterVector
from qiskit.circuit.library import QFT


def shor_circuit(N: int, a: int, measurement=True) -> QuantumCircuit:
    n = N.bit_length()          # num qubits

    # quantum register where the sequential QFT is performed
    up_qreg = QuantumRegister(n, name="up")
    # quantum register where the multiplications are made
    down_qreg = QuantumRegister(n, name="down")
    # auxiliary quantum register used in addition and multiplication
    aux_qreg = QuantumRegister(n + 2, name="aux")

    # Create Quantum Circuit
    circuit = QuantumCircuit(up_qreg, down_qreg, aux_qreg, name=f"Shor(N={N}, a={a})")

    # Create maximal superposition in top register
    circuit.h(up_qreg)

    # Initialize down register to 1
    circuit.x(down_qreg[0])

    # Apply modulo exponentiation
    modulo_power = _power_mod_N(n, N, a)
    circuit.append(modulo_power, circuit.qubits)

    # Apply inverse QFT
    iqft = QFT(len(up_qreg)).inverse().to_gate()
    circuit.append(iqft, up_qreg)

    if measurement:
        up_cqreg = ClassicalRegister(n, name="m")
        circuit.add_register(up_cqreg)
        circuit.measure(up_qreg, up_cqreg)

    return circuit


def _power_mod_N(n:int, N: int, a: int) -> Instruction:
    """Implements modular exponentiation a^x as an instruction."""
    up_qreg = QuantumRegister(n, name="up")
    down_qreg = QuantumRegister(n, name="down")
    aux_qreg = QuantumRegister(n + 2, name="aux")

    circuit = QuantumCircuit(up_qreg, down_qreg, aux_qreg, name=f"{a}^x mod {N}")

    qft = QFT(n + 1, do_swaps=False).to_gate()
    iqft = qft.inverse()

    # Create gates to perform addition/subtraction by N in Fourier Space
    phi_add_N = _phi_add_gate(_get_angles(N, n + 1))
    iphi_add_N = phi_add_N.inverse()
    c_phi_add_N = phi_add_N.control(1)

    # Apply the multiplication gates as showed in
    # the report in order to create the exponentiation
    for i in range(n):
        partial_a = pow(a, pow(2, i), N)
        modulo_multiplier = _controlled_multiple_mod_N(
            n, N, partial_a, c_phi_add_N, iphi_add_N, qft, iqft
        )
        circuit.append(modulo_multiplier, [up_qreg[i], *down_qreg, *aux_qreg])

    return circuit.to_instruction()


def _get_angles(a: int, n: int) -> np.ndarray:
    """Calculates the array of angles to be used in the addition in Fourier Space."""
    bits_little_endian = (bin(int(a))[2:].zfill(n))[::-1]

    angles = np.zeros(n)
    for i in range(n):
        for j in range(i + 1):
            k = i - j
            if bits_little_endian[j] == "1":
                angles[i] += pow(2, -k)

    return angles * np.pi


def _phi_add_gate(angles: Union[np.ndarray, ParameterVector]) -> Gate:
    """Gate that performs addition by a in Fourier Space."""
    circuit = QuantumCircuit(len(angles), name="phi_add_a")
    for i, angle in enumerate(angles):
        circuit.p(angle, i)
    return circuit.to_gate()


def _double_controlled_phi_add_mod_N(
        angles: Union[np.ndarray, ParameterVector],
        c_phi_add_N: Gate,
        iphi_add_N: Gate,
        qft: Gate,
        iqft: Gate,
    ) -> QuantumCircuit:
    """Creates a circuit which implements double-controlled modular addition by a."""
    ctrl_qreg = QuantumRegister(2, "ctrl")
    b_qreg = QuantumRegister(len(angles), "b")
    flag_qreg = QuantumRegister(1, "flag")

    circuit = QuantumCircuit(ctrl_qreg, b_qreg, flag_qreg, name="ccphi_add_a_mod_N")

    cc_phi_add_a = _phi_add_gate(angles).control(2)
    cc_iphi_add_a = cc_phi_add_a.inverse()

    circuit.append(cc_phi_add_a, [*ctrl_qreg, *b_qreg])

    circuit.append(iphi_add_N, b_qreg)

    circuit.append(iqft, b_qreg)
    circuit.cx(b_qreg[-1], flag_qreg[0])
    circuit.append(qft, b_qreg)

    circuit.append(c_phi_add_N, [*flag_qreg, *b_qreg])

    circuit.append(cc_iphi_add_a, [*ctrl_qreg, *b_qreg])

    circuit.append(iqft, b_qreg)
    circuit.x(b_qreg[-1])
    circuit.cx(b_qreg[-1], flag_qreg[0])
    circuit.x(b_qreg[-1])
    circuit.append(qft, b_qreg)

    circuit.append(cc_phi_add_a, [*ctrl_qreg, *b_qreg])

    return circuit


def _controlled_multiple_mod_N(
        n: int, N: int, a: int, c_phi_add_N: Gate, iphi_add_N: Gate, qft: Gate, iqft: Gate
) -> Instruction:

    """Implements modular multiplication by a as an instruction."""
    ctrl_qreg = QuantumRegister(1, "ctrl")
    x_qreg = QuantumRegister(n, "x")
    b_qreg = QuantumRegister(n + 1, "b")
    flag_qreg = QuantumRegister(1, "flag")

    circuit = QuantumCircuit(ctrl_qreg, x_qreg, b_qreg, flag_qreg, name="cmult_a_mod_N")

    angle_params = ParameterVector("angles", length=n + 1)
    modulo_adder = _double_controlled_phi_add_mod_N(
    angle_params, c_phi_add_N, iphi_add_N, qft, iqft
    )

    def append_adder(adder: QuantumCircuit, constant: int, idx: int):
        partial_constant = (pow(2, idx, N) * constant) % N
        angles = _get_angles(partial_constant, n + 1)
        bound = adder.assign_parameters({angle_params: angles})
        circuit.append(bound, [*ctrl_qreg, x_qreg[idx], *b_qreg, *flag_qreg])

    circuit.append(qft, b_qreg)

    # perform controlled addition by a on the aux register in Fourier space
    for i in range(n):
        append_adder(modulo_adder, a, i)

    circuit.append(iqft, b_qreg)

    # perform controlled subtraction by a in Fourier space on both the aux and down register
    for i in range(n):
        circuit.cswap(ctrl_qreg, x_qreg[i], b_qreg[i])

    circuit.append(qft, b_qreg)

    a_inv = pow(a, -1, mod=N)

    modulo_adder_inv = modulo_adder.inverse()
    for i in reversed(range(n)):
        append_adder(modulo_adder_inv, a_inv, i)

    circuit.append(iqft, b_qreg)

    return circuit.to_instruction()
