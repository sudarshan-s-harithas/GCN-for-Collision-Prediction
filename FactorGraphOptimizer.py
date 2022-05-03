import numpy as np
import torch 
import matplotlib.pyplot as plt


P = np.array( [[ 0.36433317, -0.00496474,  0.0179645,  -0.27256447],
                        [-0.07293952,  0.41734869 , 0.00587739 ,-0.24317693],
                         [ 0.15458865,  0.00744779  ,0.14326629 , 1 ]])


def GetGradsPerObs( X ):

	N1  = P[0][0]*X[0] + P[0][1]*X[1] + P[0][2]*X[2]+ P[0][3]
	D1 = P[2][0]*X[0] + P[2][1]*X[1] + P[2][2]*X[2]+ P[2][3]

	N2 = P[1][0]*X[0] + P[1][1]*X[1] + P[1][2]*X[2]+ P[1][3]
	D2 = P[2][0]*X[0] + P[2][1]*X[1] + P[2][2]*X[2]+ P[2][3]

	del_u_x = (P[0][0]*D1 - P[2][0]*N1)/(D1)**2
	del_v_x  = (P[1][0]*D2 - P[2][0]*N2)/(D2)**2


	del_u_y = (P[0][1]*D1 - P[2][1]*N1)/(D1)**2
	del_v_y = (P[1][1]*D2 - P[2][1]*N2)/(D2)**2

	del_u_z = (P[0][2]*D1 - P[2][2]*N1)/(D1)**2
	del_v_z = (P[1][2]*D2 - P[2][2]*N2)/(D2)**2


	J = np.array( [[ del_u_x , del_u_y , del_u_z] , 
		[del_v_x , del_v_y , del_v_z] ] )


	return J


def GetGrads( Predictions, Observations ,numPts ):

	# numPts = np.shape(Predictions)[0]*np.shape(Predictions)[1]

	Jacobian = np.zeros( (2*numPts , 3*numPts) )

	for i in range(numPts):
		
		val = GetGradsPerObs( Predictions[i]  )
		Jacobian[2*i:2*i+2, 3*i:3*i+3] = val

	return Jacobian



def GetProjections( predicted3Dpts , numPts):

	ImagePts = np.ones( (numPts , 4) )
	ImagePts[:, 0:3] =  predicted3Dpts
	ImagePts = ImagePts.T

	predict2d = P@ImagePts
	predict2d = predict2d[0:2 , :].T

	return predict2d 


def GetError( predictions, Observations , numPts ):

	predictions = torch.tensor(predictions).to('cpu')
	error = (predictions - Observations)
	error = np.reshape(error, (2*numPts , 1 ) )
	error_val = torch.linalg.norm(error)
	return error , float(error_val)

def GetUpdateVector( J , error ,numPts ,device ):

	

	J = torch.tensor(J).to(device)
	error = torch.tensor(error).to(device)
	I = torch.eye( 3*numPts ).to(device)

	del_X = ( torch.linalg.pinv( J.T@J + I*0.001 ) @J.T )@error 
	return del_X.to('cpu')

def OptimizePoses( predicted3Dpts,  Image2Dpts ):

	(a3,b3,c3) = np.shape(predicted3Dpts)
	(a2,b2,c2) =np.shape(Image2Dpts)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	predicted3Dpts = torch.tensor(predicted3Dpts ).to('cpu')
	Image2Dpts = torch.tensor(Image2Dpts ).to('cpu')

	iters = 5
	error_vec = [] 
	iter_x  = np.arange( 0 , iters , 1)

	for i in range( iters ):

		predicted3Dpts = torch.reshape(predicted3Dpts, ( a3*b3 , 3) ).to('cpu')
		Image2Dpts = torch.reshape(Image2Dpts, (a2*b2 , 2) ).to('cpu')

		J = GetGrads(predicted3Dpts, Image2Dpts , a3*b3)
		predict2d = GetProjections( predicted3Dpts , a3*b3)
		error , error_val = GetError( predict2d , Image2Dpts , a2*b2 )
		error_vec.append(error_val)
		del_X  = GetUpdateVector( J , error , a2*b2  ,device )
		del_X = torch.reshape( del_X , (a3,b3,c3) )

		predicted3Dpts = torch.tensor(predicted3Dpts).to('cpu')

		predicted3Dpts = torch.reshape(predicted3Dpts , (a3,b3,c3) ).to('cpu')

		predicted3Dpts -= 0.001*del_X

	# plt.plot(iter_x , error_vec )
	# plt.show()

	return predicted3Dpts.to('cpu')






