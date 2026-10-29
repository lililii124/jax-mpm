"""Reference particles for an odd-Ny grid cut by the unchanged sharp slit.

The slit is y=.5, x<=a. Cut cells have separate physical trace DOFs and
two 2x2-Gauss half-cell quadratures. The first intact cell has a positive
five-node edge enrichment so the shared crack-tip DOF is independent of
the two off-plane background coefficients. No two-row unit-damage seed is
introduced. Display vertices evaluate the exact piecewise-Q1 basis and are
not additional physical DOFs or a field symmetrization.
"""
import numpy as np


def odd_cut_reference_particles(nx,ny,crack_length=.375):
    if nx<12 or ny<13 or ny%2!=1 or abs(crack_length*nx-round(crack_length*nx))>1e-12:
        raise ValueError('Require nx>=12, odd ny>=13 and grid-aligned x tip')
    hx=1/nx;hy=1/ny;middle=ny//2;tip=int(round(crack_length*nx))
    coordinates=[(i*hx,j*hy,0.) for j in range(ny+1) for i in range(nx+1)]
    side=[0]*len(coordinates);lower={};upper={}
    for i in range(tip+1):
        lower[i]=len(coordinates);coordinates.append((i*hx,.5,0.));side.append(-1 if i<tip else 0)
        if i<tip:
            upper[i]=len(coordinates);coordinates.append((i*hx,.5,0.));side.append(1)
        else:upper[i]=lower[i]
    X=np.asarray(coordinates);side=np.asarray(side)
    points=[];nodes=[];weights=[];gradients=[];volumes=[]
    gauss=(.5-.5/np.sqrt(3),.5+.5/np.sqrt(3))
    def shape(s,t,dx,dy):
        N=np.array([(1-s)*(1-t),s*(1-t),s*t,(1-s)*t])
        G=np.array([[-(1-t)/dx,-(1-s)/dy,0],[(1-t)/dx,-s/dy,0],[t/dx,s/dy,0],[-t/dx,(1-s)/dy,0]])
        return N,G
    def add(connect,N,G,x,y,volume):
        if len(connect)==4:
            connect=list(connect)+[0];N=np.r_[N,0.];G=np.vstack((G,[0.,0.,0.]))
        points.append((x,y,0.));nodes.append(connect);weights.append(N);gradients.append(G);volumes.append(volume)
    for j in range(ny):
        for i in range(nx):
            a=j*(nx+1)+i;ordinary=[a,a+1,a+nx+2,a+nx+1]
            if j==middle and i<tip:
                quads=[[ordinary[0],ordinary[1],lower[i+1],lower[i]],
                       [upper[i],upper[i+1],ordinary[2],ordinary[3]]]
                for half,connect in enumerate(quads):
                    for s in gauss:
                        for t in gauss:
                            N,G=shape(s,t,hx,hy/2)
                            add(connect,N,G,(i+s)*hx,(j+(half+t)/2)*hy,hx*hy/8)
            elif j==middle and i==tip:
                connect=ordinary+[lower[tip]]
                # The edge shape has a kink at t=.5. Integrate EACH smooth
                # half exactly at the same Gauss order as every other cell.
                for half in range(2):
                    for s in gauss:
                        for sub_t in gauss:
                            t=(half+sub_t)/2;N,G=shape(s,t,hx,hy)
                            beta=1-abs(2*t-1);dbeta=2. if t<.5 else -2.
                            Nt=(1-s)*beta;Gt=np.array([-beta/hx,(1-s)*dbeta/hy,0.])
                            N=np.r_[N,Nt];G=np.vstack((G,Gt))
                            N[[0,3]]-=.5*Nt;G[[0,3]]-=.5*Gt
                            add(connect,N,G,(i+s)*hx,(j+t)*hy,hx*hy/8)
            else:
                for s in gauss:
                    for t in gauss:
                        N,G=shape(s,t,hx,hy)
                        add(ordinary,N,G,(i+s)*hx,(j+t)*hy,hx*hy/4)
    P=np.asarray(points);nodes=np.asarray(nodes,np.int32);W=np.asarray(weights);G=np.asarray(gradients);V=np.asarray(volumes)
    # Conforming DISPLAY subquads. Virtual intact-edge midpoints are exact
    # evaluations (.5,.5) of the corresponding physical Q1 coefficients.
    render_X=coordinates.copy();display_nodes=[[i,0] for i in range(len(X))];display_weights=[[1.,0.] for i in range(len(X))]
    render_lower=lower.copy();render_upper=upper.copy()
    for i in range(tip+1,nx+1):
        at=len(render_X);render_X.append((i*hx,.5,0.))
        display_nodes.append([middle*(nx+1)+i,(middle+1)*(nx+1)+i]);display_weights.append([.5,.5])
        render_lower[i]=render_upper[i]=at
    display_quads=[];display_lookup={}
    for j in range(ny):
        for i in range(nx):
            a=j*(nx+1)+i;quad=[a,a+1,a+nx+2,a+nx+1]
            if j==middle:
                display_lookup[(i,j,0)]=len(display_quads);display_lookup[(i,j,1)]=len(display_quads)+1
                display_quads.extend([[quad[0],quad[1],render_lower[i+1],render_lower[i]],
                                     [render_upper[i],render_upper[i+1],quad[2],quad[3]]])
            else:
                display_lookup[(i,j,0)]=len(display_quads);display_quads.append(quad)
    # Reflection distinguishes the two coincident slit faces, not coordinates
    # alone. All other reference nodes are unique geometric points.
    lookup={(round(x,12),round(y,12),int(tag)):i for i,((x,y,_),tag) in enumerate(zip(X,side))}
    mirror=np.array([lookup[(round(x,12),round(1-y,12),-int(tag))] for (x,y,_),tag in zip(X,side)],np.int32)
    plookup={tuple(np.round(x,12)):i for i,x in enumerate(P)}
    pmirror=np.array([plookup[tuple(np.round([x,1-y,z],12))] for x,y,z in P],np.int32)
    display_cell=np.array([display_lookup[(int(np.floor(x/hx)),int(np.floor(y/hy)),int(y>.5) if int(np.floor(y/hy))==middle else 0)] for x,y,_ in P],np.int32)
    seed=(abs(X[:,1]-.5)<1e-12)&(X[:,0]<=crack_length+1e-12)
    result={'coordinates':X,'particles':P,'nodes':nodes,'weight':W,'gradient':G,'volume':V,
            'triangles':np.empty((0,3),np.int32),'seed':seed,'node_side':side,
            'mirror_nodes':mirror,'mirror_particles':pmirror,'nx':nx,'ny':ny,
            'display_coordinates':np.asarray(render_X),'display_nodes':np.asarray(display_nodes,np.int32),
            'display_weights':np.asarray(display_weights),'display_quads':np.asarray(display_quads,np.int32),'display_cell_of_particle':display_cell}
    validation=validate(result)
    if not validation['passed']:raise RuntimeError('Invalid odd-grid cut reference: '+str(validation))
    result['validation']=validation
    return result


def validate(data):
    X=data['coordinates'];P=data['particles'];nodes=data['nodes'];W=data['weight'];G=data['gradient'];V=data['volume']
    mirror=data['mirror_nodes'];pmirror=data['mirror_particles'];S=np.diag([1.,-1.,1.])
    affine=np.einsum('pki,pkj->pij',X[nodes],G)
    gradient_target=np.diag([1.,1.,0.])
    order=[3,2,1,0,4]
    checks={'volume_error':float(abs(V.sum()-1)),'minimum_shape_weight':float(W.min()),
            'partition_error':float(abs(W.sum(axis=1)-1).max()),'gradient_partition_error':float(abs(G.sum(axis=1)).max()),
            'coordinate_reproduction_error':float(abs(np.einsum('pk,pki->pi',W,X[nodes])-P).max()),
            'affine_gradient_error':float(abs(affine-gradient_target).max()),
            'weight_reflection_error':float(abs(W-W[pmirror][:,order]).max()),
            'gradient_reflection_error':float(abs(G-np.einsum('pki,ij->pkj',G[pmirror][:,order],S)).max()),
            'seed_reflection_exact':bool(np.array_equal(data['seed'],data['seed'][mirror])),
            'node_reflection_involution':bool(np.array_equal(mirror[mirror],np.arange(len(X)))),
            'particle_reflection_involution':bool(np.array_equal(pmirror[pmirror],np.arange(len(P))))}
    active=(abs(W)>1e-14)|(np.linalg.norm(G,axis=2)>1e-14)
    checks['node_reflection_exact_on_active_support']=bool(np.all((mirror[nodes]==nodes[pmirror][:,order])|~active))
    checks['seed_has_zero_geometric_width']=bool(np.all(X[data['seed'],1]==.5))
    display_X=np.sum(data['display_weights'][:,:,None]*X[data['display_nodes']],axis=1)
    checks['display_coordinate_mapping_error']=float(abs(display_X-data['display_coordinates']).max())
    coefficient=np.random.default_rng(9147).normal(size=len(X))
    display_value=np.sum(data['display_weights']*coefficient[data['display_nodes']],axis=1)
    q=data['display_quads'][data['display_cell_of_particle']];corners=display_X[q]
    s=(P[:,0]-corners[:,0,0])/(corners[:,1,0]-corners[:,0,0]);t=(P[:,1]-corners[:,0,1])/(corners[:,3,1]-corners[:,0,1])
    sample_weight=np.stack(((1-s)*(1-t),s*(1-t),s*t,(1-s)*t),axis=1)
    checks['display_basis_evaluation_error']=float(abs(np.sum(W*coefficient[nodes],axis=1)-np.sum(sample_weight*display_value[q],axis=1)).max())
    checks['passed']=bool(checks['minimum_shape_weight']>=-1e-14 and all(checks[k]<1e-11 for k in ['volume_error','partition_error','gradient_partition_error','coordinate_reproduction_error','affine_gradient_error','weight_reflection_error','gradient_reflection_error','display_coordinate_mapping_error','display_basis_evaluation_error']) and checks['seed_reflection_exact'] and checks['node_reflection_involution'] and checks['particle_reflection_involution'] and checks['node_reflection_exact_on_active_support'] and checks['seed_has_zero_geometric_width'])
    return checks
