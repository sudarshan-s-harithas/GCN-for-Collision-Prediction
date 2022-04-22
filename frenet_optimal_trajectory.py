"""

Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame](https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame](https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import cubic_spline_planner
import circ

SIM_LOOP = 500

# Parameter
MAX_SPEED = 1.5*6  # maximum speed [m/s]
MAX_ACCEL = 4.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 2.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 4.0*2  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.1  # time tick [s]
MAXT = 3.0  # max prediction time [m]
MINT = 2.0  # min prediction time [m]
TARGET_SPEED = 1.5*3  # target speed [m/s]
D_T_S = 1.5*3 / 1.5  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 0.5*3 # robot radius [m]
# cost weights
KJ = 0.1
KT = 0.1
KD = 1.0
KLAT = 1.0
KLON = 1.0*5

show_animation = True

def vec_angl(x,y):
    x1=x[0]
    x2=y[0]
    y1=x[1]
    y2=y[1]
    dot = x1*x2 + y1*y2      # dot product
    det = x1*y2 - y1*x2      # determinant
    angle = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    return angle

class quintic_polynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):

        # calc coefficient of quintic polynomial
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[T**3, T**4, T**5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T**2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

        return xt


class quartic_polynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, T):

        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class Frenet_path:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []
        self.v=[]
        self.w=[]

def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0):

    frenet_paths = []

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MINT, MAXT, DT):
            fp = Frenet_path()

            lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE, TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = quartic_polynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1])**2

                tfp.cd = KJ * Jp + KT * Ti + KD * tfp.d[-1]**2
                tfp.cv = KJ * Js + KT * Ti + KD * ds
                tfp.cf = KLAT * tfp.cd + KLON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp, prev_vec):

    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            iyaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(iyaw + math.pi / 2.0)
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        prev_dx=fp.x[0]-prev_vec[0]
        prev_dy=fp.y[0]-prev_vec[1]        
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))
            fp.v.append(fp.ds[-1]/DT)
            fp.w.append(vec_angl([prev_dx,prev_dy],[dx,dy])/DT)
            prev_dx=dx
            prev_dy=dy
            	    
        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])
        fp.v.append(fp.ds[-1]/DT)
        fp.w.append(fp.w[-1])
	
        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


def check_collision(fp, ob,ob_v):

    for i in range(len(ob[:, 0])):
        d = []
        counter=0
        for (ix, iy) in zip(fp.x, fp.y):
            counter=counter+DT
            d.append((ix - (ob[i, 0]+ob_v[i,0]*counter))**2 + (iy - (ob[i, 1]+ob_v[i,1]*counter))**2)
        collision = any([di <= (2*ROBOT_RADIUS+0.2)**2 for di in d])

        if collision:
            return False

    return True

def check_collision_IVO(fp, ob,ob_v):

    for i in range(len(ob[:, 0])):
        d = []
        counter=0
        for (ix, iy) in zip(fp.x, fp.y):
            counter=counter+DT
            d.append((ix - (ob[i, 0]+ob_v[i,0]*counter))**2 + (iy - (ob[i, 1]+ob_v[i,1]*counter))**2)
        collision = any([di <= (2*ROBOT_RADIUS)**2 for di in d])

        if collision:
            return False

    return True



def DrawPaths( Xpts, Ypts ):

    plt.plot( Xpts ,Ypts)
    plt.show()




def getPrior( d ):

    w = [ 0.25 , 0.6 , 0.15 ]

    L_avail = [ 0, 1 , 2]

    W = np.zeros(( d, 3))

    L = []

    for i in range(d):

        initLabel = np.random.choice( L_avail)
        L.append( initLabel )


    for i in range(d):

        W[i] = w


    return W, L 



def GetGaussian( x,  mean , sd):
    # sd = np.sqrt(sd)
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density



def PerformNBP(   GMMs ):

    ## inpput format
    ## the input consists of d GMM each with M gaussian functions within 
    ## in this case the value of d is a variable but we fix M =3 
    ## the input is a list of shape ( d , 6) , each row corresponds to a new GMMM whereas
    ## the first 3 coloums are the means of each guassian and the last 3 coloumns are the corresponding variances 


    d = len(GMMs)
    w, L = getPrior(d)
    W_update = w
    iters = 5


    for iterate in range( iters ):

        # print( "Enter Iter " + str(iterate))

        if( d > 1 ):

            for j in range(d):

                inv_sum_ =0

                var_= 0 
                mean_temp = 0 

                for j2 in range(d):

                    if( j2 != j ):
                        inv_sum_ += 1/GMMs[j][ 3+L[j2] ]
                    # print( inv_sum_ )
                    # print("******")


                inv_var_star = inv_sum_


                for j2 in range( d ):

                    if( j2 != j ):

                        mean_temp += (1/GMMs[j][ 3+L[j2] ])*GMMs[j][ L[j2] ]


                mean_star = (1/inv_sum_)*mean_temp

                inv_sum2 = 0
                mean_temp2 = 0

                x = GMMs[j][1]


                for i in range(3):

                    inv_sum2 = (inv_var_star)+ ( 1/ GMMs[j][2+i] )
                    mean_temp2 =  inv_sum2*( (inv_var_star*mean_star) +  ( inv_sum2*GMMs[j][i] ) )

                    # print(  GetGaussian( mean_temp2 , mean_temp2 , (1/inv_sum2) )  )

                    W_update[j][i] = W_update[j][i] *( GetGaussian( x , mean_star , (inv_var_star) )*   GetGaussian( x , GMMs[j][i] , GMMs[j][3+i] ) )/( GetGaussian( x , mean_temp2 , (inv_sum2) ) )


                W_update[j] = W_update[j] / np.sum(W_update[j])
                L[j] = np.argmax( W_update[j]  )

            s_temp = 0 
            m_temp = 0


            for i in range(d):

                M = GMMs[ i][L[i]]
                S = GMMs[ i][L[i] + 3]

                s_temp += 1/S 
                m_temp +=  (1/S )*M


            sigma_final = 1/ s_temp
            mean_final = sigma_final*m_temp

            xt = np.random.normal(mean_final, np.sqrt(sigma_final), 3 )

            means = []
            var_vals = []
            weights = []

            for i in range(3):

                x = np.random.normal(xt[i],  1.0, 1)[0]
                means.append(x)
                var_vals.append(1)
                weights.append(0.33)

            print( means )
            print(var_vals)
            print(weights)
            print("************")
            return means , var_vals,  weights 


        # if( d == 1 ):






def GetGMMs( Xpts , Ypts  , obs ):

    GMMS_dist = []
    numPts = np.shape(Xpts)[0]
    numTrajs = np.shape(Xpts)[1]

    for i in range( numTrajs ):
        for j in range( numPts ):

        

            for o in range( len( obs )):

                Pt = np.array( [ Xpts[j][i] , Ypts[j][i] ] )
                distance = np.linalg.norm( Pt - obs[o])


                if( distance < 2 ):

                    dist = [ distance -0.2 , distance , distance+0.2 , (0.7)**2 , (0.8)**2 , (0.9)**2  ]
                    GMMS_dist.append( dist )

    return GMMS_dist


def CheckCollisionGNN( paths , obs ):

    numPaths = len(paths)
    BatchSize = 10
    numBatches = numPaths/BatchSize 


    for i in  range( int(numBatches)):

        BatchPaths = paths[ i*BatchSize:i*BatchSize+ BatchSize ]

        Xpts = np.zeros( ( 20 , 10 ) )
        Ypts = np.zeros( ( 20 , 10 ) )

        cnt = 0 

        for p in BatchPaths:

            # DrawPaths( p.x , p.y )
            Xpts[:, cnt ] = p.x[0:20]
            Ypts[:, cnt ] = p.y[0:20]

            cnt += 1

        GMMS_dist = GetGMMs(Xpts  , Ypts, obs)
        PerformNBP( GMMS_dist )






def check_paths(fplist, ob, ob_v):

    okind = []
    CheckCollisionGNN( fplist  , ob   )
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
            continue
        #print(len(ob))    
        if(len(ob)>0):
            if not check_collision(fplist[i], ob,ob_v):
                continue

        okind.append(i)

    return [fplist[i] for i in okind]


def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob,ob_v, prev_vec):

    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
    #print(len(fplist))
    fplist = calc_global_paths(fplist, csp,prev_vec)
    #print(len(fplist))
    fplist = check_paths(fplist, ob, ob_v)
    #print(len(fplist))
    # find minimum cost path
    mincost = float("inf")
    bestpath = None
    for fp in fplist:
        if mincost >= fp.cf:
            mincost = fp.cf
            bestpath = fp

    return bestpath


def generate_target_course(x, y):
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


def main():
    print(__file__ + " start!!")

    # way points
    wx = [0.0, 10.0, 20.5, 35.0, 70.5]
    wy = [0.0, -6.0, 5.0, 6.5, 0.0]
    # obstacle lists
    ob = np.array([[20.0, 10.0],
                   [30.0, 6.0],
                   [30.0, 8.0],
                   [35.0, 8.0],
                   [50.0, 3.0]
                   ])
             

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)
    wx=np.array(wx)
    wy=np.array(wy)  
    # initial state
    c_speed = 10.0 / 3.6  # current speed [m/s]
    c_d = 2.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current latral acceleration [m/s]
    s0 = 0.0  # current course position

    area = 20.0  # animation area length [m]

    for i in range(SIM_LOOP):
        path = frenet_optimal_planning(
            csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob)

        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(tx,ty) 
            plt.text(wx[0]-0.3, wy[0]-0.3,'initial position')
            circ.circ(wx[0],wy[0],0.2 ,color='green')
            plt.text(wx[4]-0.3, wy[4]-0.3,'final position')
            circ.circ(wx[4],wy[4],0.2 ,color='green')
            for z in range(1,len(wx)-1):
             plt.text(wx[z]-0.3, wy[z]-0.3,'generated waypoints')
             circ.circ(wx[z],wy[z],0.2 ,color='green')
            for z in range(len(ob)):
             circ.circ(ob[z, 0],ob[z, 1],0.5 ,color='red')
             plt.text(ob[z, 0]+0.6, ob[z, 1]+0.6,'Obstacle')
            #plt.plot(path.x[1:], path.y[1:], "-or")
            circ.circ(path.x[1], path.y[1], 1,color='blue')
            plt.text(path.x[1]+1.2, path.y[1]+1.2,'End Effector')
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.savefig('outputs/dist-{}.png'.format(str(i).zfill(4)), dpi=300)
            plt.pause(0.0001)

    print("Finish")
    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main()
