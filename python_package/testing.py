import numpy as np
import matplotlib.pyplot as plt 
from inspect import isfunction
from qiskit import QuantumCircuit
from qiskit import transpile  
from qiskit.quantum_info import Statevector, Operator
from qiskit.quantum_info import hellinger_distance
from qiskit.quantum_info import SparsePauliOp, process_fidelity
from scipy.linalg import expm
from qiskit_ibm_runtime.fake_provider import FakeBurlingtonV2 as FakeDevice
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA


### NO MODIFICAR ###

def test_1a( qc_ghz_op : QuantumCircuit ):

    qc_ghz_op = transpile( qc_ghz_op )
    n_qubits = 4 
    qc_ghz = QuantumCircuit( n_qubits ) 
    qc_ghz.h(0)
    for j in range(n_qubits-1):
        qc_ghz.cx(j,j+1)
    state = Statevector( qc_ghz )
    state_op = Statevector( qc_ghz_op )

    if not isinstance( qc_ghz_op, QuantumCircuit ):
        print('No es un circuito cuántico')
    elif qc_ghz_op.num_qubits != n_qubits:
        print('El circuito no tiene 4 qubits')
    elif not np.isclose( np.linalg.norm( state-state_op ), 0 ):
        print('El circuito no prepara un estado GHZ')
    elif not ( qc_ghz_op.depth() == 3  ):
        print('La profundidad del circuito es muy grande')
    else:
        print('Felicidades, tu solución es correcta!')

def test_1b( qc_ghz_op ):

    n_qubits = 4 
    qc_ghz_device = QuantumCircuit( n_qubits ) 
    qc_ghz_device.h(0)
    for j in range(n_qubits-2):
        qc_ghz_device.cx(j,j+1)

    qc_ghz_device.cx(2,1)
    qc_ghz_device.cx(1,2)
    qc_ghz_device.cx(2,1)
    qc_ghz_device.cx(1,3)

    device_backend = FakeDevice() 
    simulator_noise = AerSimulator.from_backend(device_backend) 
    qc_ghz_device_measured = qc_ghz_device.copy() 
    qc_ghz_device_measured.measure_all() 
    qc_ghz_device_measured =  transpile( qc_ghz_device_measured, 
                                        device_backend, optimization_level=0 ) 

    counts_ideal = { '0000':500, '1111':500  }

    qc_ghz_op_measured = qc_ghz_op.copy() 
    qc_ghz_op_measured.measure_all() 
    qc_ghz_op_measured =  transpile( qc_ghz_op_measured, device_backend, optimization_level=0 ) 

    state = Statevector( qc_ghz_device )
    state_op = Statevector( qc_ghz_op )

    if not isinstance( qc_ghz_op, QuantumCircuit ):
        print('No es un circuito cuántico')
    elif qc_ghz_op.num_qubits != n_qubits:
        print('El circuito no tiene 4 qubits')
    elif not np.isclose( np.linalg.norm( state-state_op ), 0 ):
        print('El circuito no prepara un estado GHZ')
    elif not ( qc_ghz_op_measured.depth() == 5  ):
        print('La profundidad del circuito mapeado al circuito es muy grande')
    else:
        error1 = 0
        error2 = 0
        for _ in range(100):
            counts_op = simulator_noise.run( qc_ghz_op_measured ).result().get_counts() 
            counts_device = simulator_noise.run( qc_ghz_device_measured ).result().get_counts() 

            error1 += hellinger_distance(counts_ideal, counts_device)
            error2 += hellinger_distance(counts_ideal, counts_op)

        if error2 < error1:
            print( 'Felicidades, tu solución es correcta!' )
        else:
            print( 'El error de tu circuito es mayor!')

def test_2a( Fourier ):

    sol = False 
    if not isfunction( Fourier ):
        print('Input no es una función')
    else:
        for num_qubits in range(2,6):

            F = np.exp( 2j*np.pi*np.outer( np.arange(2**num_qubits), np.arange(2**num_qubits) )/2**num_qubits  ) / np.sqrt(2**num_qubits)

            qc = Fourier( num_qubits )
            if not np.isclose( np.linalg.norm( F-Operator(qc).to_matrix() ), 0 ) :
                sol = False 
                print( 'La función no implementa la transformada de Fourier para {} qubits'.format(num_qubits) )
                break
            else:
                sol = True 

    if sol: 
        print('Felicidades, tu solución es correcta!')


def test_2b( U_to_n ):
    sol = False 
    for power in range(1,6):
        U1 = np.diag([1,1,1,np.exp(power*1j*2*np.pi*0.375)])
        U2 = Operator( U_to_n(power) ).to_matrix() 
        if not np.isclose( np.linalg.norm( U1-U2 ), 0 ) :
            sol = False 
            print( 'La función no implementa '+r'$U^n$'+' para potencia {}'.format(power) )
            break
        else:
            sol = True 

    if sol: 
        print('Felicidades, tu solución es correcta!')


def test_2c( QuantumPhaseEstimation ):

    sol = False
    for num_qubits in range(3,6):
        phi = 0.375 
        qc = QuantumPhaseEstimation(num_qubits)
        backend = AerSimulator()
        job = backend.run( qc )
        counts = job.result().get_counts()
        phi_hat = int( max(counts ), 2 ) / 2**num_qubits
        if not np.isclose( np.abs(phi_hat-phi), 0 ) :
            sol = False 
            print( 'La función no estima correctamente la fase para {} qubits'.format(num_qubits) )
            print( r'$\tilde\phi=$'+'{}'.format(phi_hat))
            break
        else:
            sol = True 

    if sol: 
        print('Felicidades, tu solución es correcta!')

####################################

def test_3a( Aj, Bk ):

       A = np.zeros((3,2,2), dtype=complex)
       B = np.zeros((3,2,2), dtype=complex)

       A[0] = np.array([[1.+0.j, 0.+0.j],
              [0.+0.j, 1.+0.j]])
       A[1] = np.array([[ 0.92387953+0.j,  0.38268343+0.j],
              [-0.38268343+0.j,  0.92387953+0.j]])
       A[2] = np.array([[ 0.70710678+0.j,  0.70710678+0.j],
              [-0.70710678+0.j,  0.70710678+0.j]])
       B[0] = np.array([[ 0.92387953+0.j,  0.38268343+0.j],
              [-0.38268343+0.j,  0.92387953+0.j]])
       B[1] = np.array([[ 0.70710678+0.j,  0.70710678+0.j],
              [-0.70710678+0.j,  0.70710678+0.j]])
       B[2] = np.array([[ 0.38268343+0.j,  0.92387953+0.j],
              [-0.92387953+0.j,  0.38268343+0.j]])
       
       sol = True
       for j, a in enumerate(Aj):
              if a.num_qubits > 1:
                     print('Los circuitos deben tener 1 qubit')
              if np.isclose( np.linalg.norm( Operator(a).to_matrix() - A[j] ), 0 ):
                     pass
              else:
                     sol = False 
                     print('El circuito {} de Alice no es correcto'.format(j))

       for k, b in enumerate(Bk):
              if b.num_qubits > 1:
                     print('Los circuitos deben tener 1 qubit')
              if np.isclose( np.linalg.norm( Operator(b).to_matrix() - B[k] ), 0 ):
                     pass
              else:
                     sol = False
                     print('El circuito {} de Bob no es correcto'.format(k) )

       if sol:
              print('Felicitaciones, tu solución es correcta!')

def test_3b( qcs, alice_trits, bob_trits ):
    Ops_2qb = np.array([[[ 2.70598050e-01+0.j,  2.70598050e-01+0.j,  6.53281482e-01+0.j,
          6.53281482e-01+0.j],
        [-6.53281482e-01+0.j,  6.53281482e-01+0.j, -2.70598050e-01+0.j,
          2.70598050e-01+0.j],
        [ 6.53281482e-01+0.j,  6.53281482e-01+0.j, -2.70598050e-01+0.j,
         -2.70598050e-01+0.j],
        [ 2.70598050e-01+0.j, -2.70598050e-01+0.j, -6.53281482e-01+0.j,
          6.53281482e-01+0.j]],

       [[ 5.00000000e-01+0.j,  5.00000000e-01+0.j,  5.00000000e-01+0.j,
          5.00000000e-01+0.j],
        [-5.00000000e-01+0.j,  5.00000000e-01+0.j, -5.00000000e-01+0.j,
          5.00000000e-01+0.j],
        [ 5.00000000e-01+0.j,  5.00000000e-01+0.j, -5.00000000e-01+0.j,
         -5.00000000e-01+0.j],
        [ 5.00000000e-01+0.j, -5.00000000e-01+0.j, -5.00000000e-01+0.j,
          5.00000000e-01+0.j]],

       [[ 6.53281482e-01+0.j,  6.53281482e-01+0.j,  2.70598050e-01+0.j,
          2.70598050e-01+0.j],
        [-2.70598050e-01+0.j,  2.70598050e-01+0.j, -6.53281482e-01+0.j,
          6.53281482e-01+0.j],
        [ 2.70598050e-01+0.j,  2.70598050e-01+0.j, -6.53281482e-01+0.j,
         -6.53281482e-01+0.j],
        [ 6.53281482e-01+0.j, -6.53281482e-01+0.j, -2.70598050e-01+0.j,
          2.70598050e-01+0.j]],

       [[ 2.77555756e-17+0.j,  5.00000000e-01+0.j,  5.00000000e-01+0.j,
          7.07106781e-01+0.j],
        [-7.07106781e-01+0.j,  5.00000000e-01+0.j, -5.00000000e-01+0.j,
          2.77555756e-17+0.j],
        [ 7.07106781e-01+0.j,  5.00000000e-01+0.j, -5.00000000e-01+0.j,
         -2.77555756e-17+0.j],
        [ 2.77555756e-17+0.j, -5.00000000e-01+0.j, -5.00000000e-01+0.j,
          7.07106781e-01+0.j]],

       [[ 2.70598050e-01+0.j,  6.53281482e-01+0.j,  2.70598050e-01+0.j,
          6.53281482e-01+0.j],
        [-6.53281482e-01+0.j,  2.70598050e-01+0.j, -6.53281482e-01+0.j,
          2.70598050e-01+0.j],
        [ 6.53281482e-01+0.j,  2.70598050e-01+0.j, -6.53281482e-01+0.j,
         -2.70598050e-01+0.j],
        [ 2.70598050e-01+0.j, -6.53281482e-01+0.j, -2.70598050e-01+0.j,
          6.53281482e-01+0.j]],

       [[ 5.00000000e-01+0.j,  7.07106781e-01+0.j,  5.55111512e-17+0.j,
          5.00000000e-01+0.j],
        [-5.00000000e-01+0.j,  5.55111512e-17+0.j, -7.07106781e-01+0.j,
          5.00000000e-01+0.j],
        [ 5.00000000e-01+0.j,  5.55111512e-17+0.j, -7.07106781e-01+0.j,
         -5.00000000e-01+0.j],
        [ 5.00000000e-01+0.j, -7.07106781e-01+0.j, -5.55111512e-17+0.j,
          5.00000000e-01+0.j]],

       [[-2.70598050e-01+0.j,  6.53281482e-01+0.j,  2.70598050e-01+0.j,
          6.53281482e-01+0.j],
        [-6.53281482e-01+0.j,  2.70598050e-01+0.j, -6.53281482e-01+0.j,
         -2.70598050e-01+0.j],
        [ 6.53281482e-01+0.j,  2.70598050e-01+0.j, -6.53281482e-01+0.j,
          2.70598050e-01+0.j],
        [-2.70598050e-01+0.j, -6.53281482e-01+0.j, -2.70598050e-01+0.j,
          6.53281482e-01+0.j]],

       [[ 0.00000000e+00+0.j,  7.07106781e-01+0.j,  0.00000000e+00+0.j,
          7.07106781e-01+0.j],
        [-7.07106781e-01+0.j,  0.00000000e+00+0.j, -7.07106781e-01+0.j,
          0.00000000e+00+0.j],
        [ 7.07106781e-01+0.j,  0.00000000e+00+0.j, -7.07106781e-01+0.j,
          0.00000000e+00+0.j],
        [ 0.00000000e+00+0.j, -7.07106781e-01+0.j,  0.00000000e+00+0.j,
          7.07106781e-01+0.j]],

       [[ 2.70598050e-01+0.j,  6.53281482e-01+0.j, -2.70598050e-01+0.j,
          6.53281482e-01+0.j],
        [-6.53281482e-01+0.j, -2.70598050e-01+0.j, -6.53281482e-01+0.j,
          2.70598050e-01+0.j],
        [ 6.53281482e-01+0.j, -2.70598050e-01+0.j, -6.53281482e-01+0.j,
         -2.70598050e-01+0.j],
        [ 2.70598050e-01+0.j, -6.53281482e-01+0.j,  2.70598050e-01+0.j,
          6.53281482e-01+0.j]]])

    is_equal = False
    op_indices = []
    for j, qc in enumerate(qcs):
        qc = qc.copy()
        qc.remove_final_measurements()
        is_equal = False
        for ind, op in enumerate(Ops_2qb):
            if np.isclose(np.linalg.norm(Operator(qc).to_matrix() - op), 0):
                is_equal = True
                op_indices.append(ind)
        if not is_equal:
            print('El circuito {} está incorrecto'.format(j))
            break

    op_count = [op_indices.count(ind) for ind in range(len(Ops_2qb))]
    trit_combinations = []
    for aa, bb in zip(alice_trits, bob_trits):
        trit_combinations.append(int(str(aa) + str(bb), base=3))
    ideal_count = [trit_combinations.count(index) for index in range(9)]

    counts_correct = set(ideal_count) == set(op_count)
    if not counts_correct:
        return print('El muestreo sobre los trits elegidos no es correcto')
    if is_equal:
        return print('Felicitaciones, tu solución es correcta!')


def test_3c( key, alice_random_trits, bob_random_trits  ):
    len_key = 0
    num_trits = len(alice_random_trits)
    for j in range(num_trits):
        a = alice_random_trits[j]
        b = bob_random_trits[j]
        if (a==1 and b==0) or (a==2 and b==1):
            len_key = len_key + 1 

    if len(key) == len_key:
        print('Felicidades, tu clave es segura')
    else:
        print('La longitud de tu clave es incorrecta')

def test_4( U_trotterize ):
    op_list = []
    num_qubits = 5
    for k in range(num_qubits-1):
        XX = num_qubits * ['I']
        XX[ k ] = 'X'
        XX[ k+1 ] = 'X'
        XX = "".join(XX)  
        
        YY = num_qubits * ['I']
        YY[ k ] = 'Y'
        YY[ k+1 ] = 'Y'
        YY = "".join(YY)  

        ZZ = num_qubits * ['I']
        ZZ[ k ] = 'Z'
        ZZ[ k+1 ] = 'Z'
        ZZ = "".join(ZZ) 
        
        op_list.append( (XX,1) )
        op_list.append( (YY,1) )
        op_list.append( (ZZ,1) )

    H = SparsePauliOp.from_list( op_list )

    def U_mh(t):
        return expm( -1j*H.to_matrix()*t )

    t_target = 1
    U_target = Operator(U_mh(t_target))
    m = 5
    U_trotter = Operator( 
                U_trotterize(t_target/m, trotter_steps=m) )
    fidelity = process_fidelity(U_trotter, target=U_target)
    
    print('Fidelidad=', fidelity )
    if fidelity >= 0.9:
        print('Felicidades, su solución tiene una fidelidad superior al 90%')
    else:
        print('Su solución tiene fidelidad muy baja.')

#####################################
def test_5a( folding ):
    sol = True
    for num_qubit in [1,2,3]:
        qc_U = random_circuit(num_qubit,4)
        for N in [0,1,2,3]:
            qc_U_N = folding( qc_U, N )
            if qc_U.depth()*(2*N+1)==qc_U_N.depth():
                pass
            else:
                sol = False
                break
            if np.isclose(Operator( qc_U ).to_matrix(), 
                            Operator( qc_U_N ).to_matrix()).all():
                pass
            else:
                sol=False
                break 
    if sol:
        print('Felicitaciones, tu solución es correcta!')
    else:
        print('Su solución está equivocada, intenta de nuevo')

def test_5b( A ):
    A_th = np.array([[ 0.+0.j,  3.+0.j,  0.+0.j,  0.-3.j],
                [ 3.+0.j,  0.+0.j,  0.-1.j,  0.+0.j],
                [ 0.+0.j,  0.+1.j,  0.+0.j, -3.+0.j],
                [ 0.+3.j,  0.+0.j, -3.+0.j,  0.+0.j]])
    
    if str(type(A)) == "<class 'qiskit.quantum_info.operators.symplectic.sparse_pauli_op.SparsePauliOp'>" :
        if np.isclose( A, A_th ).all():
            print('Felicitaciones, tu solución es correcta!')
        else:
            print('Su solución está equivocada, intenta de nuevo')
    else:
        print('A tiene que ser un operador SparsePauliOp')

from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_aer.primitives import Estimator
from qiskit.primitives import Estimator as Estimator_ideal

def test_5c( extrapolation, A, Ns, folding ):

    qc_U_1 = QuantumCircuit(2)
    qc_U_1.h(0)
    qc_U_1.cx(0,1)
    qc_U_1.sdg(1)

    qc_U_2 = QuantumCircuit(2)
    qc_U_2.h(0)
    qc_U_2.cx(0,1)
    qc_U_2.sdg(1)

    sol = True
    
    for error in [0.1, 0.01, 0.001]:

        noise_model = NoiseModel()
        error = depolarizing_error( 0.01, 1 )
        noise_model.add_quantum_error( error, ['x', 'h', 'u', 'y', 'z'], [0] )
        noise_model.add_quantum_error( error, ['x', 'h', 'u', 'y', 'z'], [1] )

        backend = Estimator( backend_options={'noise_model':noise_model},
                            run_options={'shots':100000,
                                        'seed':0 },
                            skip_transpilation = True ) 
        
        backend2 = Estimator_ideal() 

        for qc_U in [qc_U_1, qc_U_2]:
            
            obs = []
            for n in Ns:
                qc_U_N = folding( qc_U, n )
                job = backend.run( qc_U_N, A )
                obs.append( job.result().values[0] )

            obs_ideal = backend2.run( qc_U, A ).result().values[0]

            a, b = extrapolation( Ns, obs )
            obs_fit = a * (2*np.array(Ns)+1) + b 
            error_fit = np.sum( (np.array(obs)-obs_fit) )

            if error_fit>0.01:
                print('Su solución está equivocada, intenta de nuevo.')
                sol = False
                break

            # print( obs_ideal, b )
            if np.abs( obs_ideal - b) < 0.09 :
                pass
            else:
                print('Su solución está equivocada, intenta de nuevo.')
                sol = False
                break  
        if not sol:
            break
            
    if sol:
        print('Tu solución esta correcta!')   

######################################3

def test_6a(H_Schwinger):

    def matrix_th_2q(m):
        M = np.diag([0,-m,m,0])
        M[1,2] = M[2,1] = 2 
        return M

    sol = True
    for m in range(-10,10, 20):
        H = H_Schwinger( 2, m )
        if not np.isclose( H.to_matrix(), matrix_th_2q(m) ).all():
            print('El Hamiltoniano esta incorrecto')
            sol = False
            break
    if sol:
        print('Felicidades, tu Hamiltoniano esta correcto!')



def test_6b(H_Schwinger,var_ansatz):
    
    sol = True
    if var_ansatz.num_parameters == 0 :
        print('Tu circuito no tiene parametros.')
        sol = False 

    for op in var_ansatz.count_ops().keys():
        if op in ['ry', 'cx']:
            pass
        else:
            print('Estas usando una puerta no permitida.')

    def VQE4Schwinger( m, plot=True ):

        if var_ansatz.num_parameters == 0 :
            E = 0
        else:
            np.random.seed(0)
            H = H_Schwinger( 4, m )
            optimizer = COBYLA(maxiter=500)

            counts = []
            values = []
            def store_intermediate_result(eval_count, parameters, mean, std):
                counts.append(eval_count)
                values.append(mean)

            quantum_solver = VQE( Estimator(), var_ansatz, optimizer, 
                                    initial_point = 0.1*np.ones(var_ansatz.num_parameters), 
                                    callback=store_intermediate_result )
            result = quantum_solver.compute_minimum_eigenvalue(operator=H)
            E = result.eigenvalue.real 

            if plot:
                plt.plot( counts, values )
                plt.xlabel('Evaluaciones')
                plt.ylabel('Energía')

        return E

    ms =  np.linspace( -5, 5, 21) 

    E_np = []
    for m in ms:
        H = H_Schwinger( 4 , m )
        vals, vecs = np.linalg.eigh( H.to_matrix() )
        E_np.append(vals[0])
    E_np = np.array(E_np)

    E_vs_m = [ VQE4Schwinger(m, False) for m in ms ]
    plt.plot( ms, E_vs_m, '-o' )
    plt.plot( ms, E_np )
    plt.xlabel('masa')
    plt.ylabel('Energía')
    plt.legend(['VQE', 'Energía Mínima'])

    if np.linalg.norm( E_vs_m - E_np ) > 2:
        print('Tu circuito variación no es suficientemente expresivo para encontrar la solución')
        sol = False 

    if sol:
        print('Tu solución esta correcta!')


def test_6c( H_Schwinger, CVQE4Schwinger, VQE4Schwinger ):
    ms =  np.linspace( -5, 5, 21) 

    E_np = []
    for m in ms:
        H = H_Schwinger( 4 , m )
        vals, vecs = np.linalg.eigh( H.to_matrix() )
        E_np.append(vals[:2])
    E_np = np.array(E_np)

    E_vs_m = [ VQE4Schwinger(m, False) for m in ms ]
    E0_plus_E1_vs_m = 2*np.array([ CVQE4Schwinger(m, False) for m in ms ])
    E1_vs_m = np.array(E0_plus_E1_vs_m) - np.array(E_vs_m)
    plt.plot( ms, E_vs_m, ':o', color='tab:blue' )
    plt.plot( ms, E1_vs_m, ':o', color='tab:orange' )
    plt.plot( ms, E_np[:,0], color='tab:blue' )
    plt.plot( ms, E_np[:,1], color='tab:orange' )
    plt.xlabel('masa')
    plt.ylabel('Energía')
    plt.legend([ 'Basal VQE', 'Excitado CVQE', 
                'Basal Exacto', 'Excitado Exacto' ]) 

    if np.mean( np.abs( E_np - np.array([ E_vs_m, E1_vs_m ]).T )**2 ) > 0.7:
        print('Tu solución esta correcta')
    else:
        print('Tu solución esta incorrecta.')


def test_7a(qc_function: callable):
    qc = qc_function()
    sol = True

    if qc.num_qubits != 4:
        sol = False
        print('Tu circuito no tiene 4 qubits.')

    if qc.num_parameters == 0:
        print('Tu circuito no tiene parametros.')
        sol = False
    elif qc.num_parameters > 4:
        print('Tu circuito tiene muchos parametros')
        sol = False

    if qc.depth() >= 20:
        print(f'Tu circuito debe tener una profundidad menor a 20. Tiene {qc.depth()}')

    for op in qc.count_ops().keys():
        if op in ['h', 'rz', 'cx', 'ry']:
            pass
        else:
            print(f'Estas usando una puerta no permitida, {op}.')

    if sol:
        print('Tu circuito es correcto')
