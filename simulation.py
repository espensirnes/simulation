#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import functions as fu
from scipy import special
from scipy import stats
from scipy.stats import binom
import traceback
import sys
import calculations



def run_sims_window(win,sd,periods,d,nsims,adj,mixed_norm,cluster,p,sd_arr,sds,mixedcovadj):

	arr=[]
	sd_arr=sd_arr*sd
	pmsq=np.sum(p*sd_arr**2)**0.5
	psd=np.sum(p*sd_arr)
	
	smean=np.mean(sds)
	#avg_tz_bias,void=calc_adj(win, cluster, pmsq*(smean/sd),d,p,sd_arr*(smean/sd))

	#for i in [1, 2, 4, 8, 15, 30, 60, 120, 240, 480, 960, 1920, 3600, 7200, 14400, 28800]:
	#	print(cluster_inflation(p,i,cluster,periods,sd_arr))

	#(ineff_c_abs, ineff_c_sq) =cluster_inflation(p,win,cluster,periods,sd_arr,d)

	tz_bias,E_abs,Es1,Es2,Es4=calc_adj(win, cluster, pmsq, d, p, sd_arr)
	E_rng=calc_E_rng(win, cluster, pmsq, d,psd, p, sd_arr)

	v,vabs=analytical_efficiency(periods,d,sd,psd,p,cluster,sd_arr,mixedcovadj,win, tz_bias,Es1,Es2,Es4)
	
	for i in range(nsims):
		r=simulation(1E-10, sd, periods, d, win,adj,mixed_norm,cluster,p,sd_arr,E_abs,E_rng,pmsq,psd,1,tz_bias)
		arr.append(r)	
	arr=np.array(arr)
	m=np.mean(arr,0)

	abserr=np.mean(np.abs((arr-m)/m),0)/(2/np.pi)**0.5	#np.mean(np.abs(arr-m),0)/((2/np.pi)**0.5)

	m_arr=np.concatenate((m-1,abserr,np.abs(m-1),
	                      np.abs([v,vabs,E_rng/psd-1,tz_bias-1,E_abs/psd-1])))
	m_arr=m_arr.reshape((len(m_arr),1))
	print((abserr,v,vabs))
	return m_arr


def analytical_efficiency(T,d,sd,psd,p,cluster,sd_arr,mixedcovadj,win,tz_bias,Es1,Es2,Es4):
	if d==0:
		d=1

	windows=int(T/win)

	
	Pd=((2/np.pi)**0.5*psd/d)
	tz2bias0=Pd*d**2

	var,c,pdi,tzi=analeff(T,d,p,sd_arr,mixedcovadj,cluster)


	tzi=sd*((2/np.pi)**0.5*d)
	clusterineff=cluster*(tzi**2-tz2bias0**2)/T
	ineff=var+clusterineff
	inefficiency=ineff/tz2bias0**2
	

	#adding intrinsic inefficiency
	intr=((2*Es4+cluster*(Es4-Es2**2))/Es2**2)/windows
	intrabs=(((0.5*np.pi-1)*Es2+cluster*(Es2-Es1**2))/Es1**2)/windows
	
	intr=2/windows
	intrabs=(0.5*np.pi-1)/windows
	
	ret=(inefficiency+intr)**0.5
	retabs=(inefficiency+intrabs)**0.5
	return ret,retabs

def analeff(T,d,p,sd_arr,mixedcovadj,cluster):
	
	covprob,k,Pd=calculations.get_covprobarray(T,p,sd_arr,mixedcovadj,d,cluster)
	
	
	
	n_diagonals=np.arange(T-1,0,-1)
	tz2bias0=Pd*d**2
						
	P=np.sum(2*n_diagonals[:k]*covprob)
	P+=np.sum(2*n_diagonals[k:]*(Pd**2))
	P+=T*Pd
	
	P=P/(T**2)
	var=P*d**4-tz2bias0**2
	return var,covprob,Pd,tz2bias0

	

def calc_adj(win,cluster,pmsq,d,p,sd_arr):
	P=1
	var=pmsq**2
	if win==0:
		return 1,np.sum(p*sd_arr)
	if cluster>1:
		if win>cluster:

			wc=win/cluster
			wci=int(wc)
			var1,P1=prob_space(p, sd_arr, wci+1)
			var2,P2=prob_space(p, sd_arr, wci+2)
			tz1=ticksizebias(var1*(win/(wci+1)), P1, win, pmsq, d)
			tz2=ticksizebias(var2*(win/(wci+2)), P2, win, pmsq, d)
			E1=E_abs_calc(var1,P1,wci+1)
			E2=E_abs_calc(var2,P2,wci+2)
			x=(wc-wci)
			E = (1-x)*E1+x*E2
			tz = (1-x)*tz1+x*tz2
		else:
			var,P=prob_space(p, sd_arr, 1)
			tz=ticksizebias(var*win, P, win, pmsq, d)
			E=E_abs_calc(sd_arr**2,p,1)
	else:
		var,P=prob_space(p, sd_arr, win)
		tz=ticksizebias(var, P, win, pmsq, d)
		E = E_abs_calc(var,P,win)
	Es1=np.sum(P*var**0.5)
	Es2=np.sum(P*var)
	Es4=np.sum(P*var**2)
	return tz,E,Es1,Es2,Es4

def calc_E_rng(win, cluster, pmsq, d,psd, p, sd_arr):
	if win<2:
		return psd
	tz,E,Es1,Es2, Es4 =calc_adj(win-1, cluster, pmsq, d, p, sd_arr)
	a=np.exp(-(win-2)**0.5*0.267)
	return E#(E*(2-a))	

def ticksizebias(var,P,win,pmsq,d):
	tz=asympt_var(d, var)
	tz=np.sum(P*tz)**0.5/(pmsq*win**0.5)
	return tz

def asympt_var(d,var):

	sd=var**0.5
	if d==0:
		v=1
	v1=var+(d**2)/6.0
	v2=2*(d*sd)/((2.0*np.pi)**0.5)
	cond=d/sd<2.5
	v=cond*v1+(cond==0)*v2
	return v

def cluster_inflation(p,win,cluster,T,sd_arr,d):
	#if cluster==1:
	#	return 1

	var=np.sum(p*sd_arr**2)
	sd=np.sum(p*sd_arr)


	N=int(T/win)


	Es4,E_abs_sq_inv=Esd4(p, sd_arr, win,1,var)

	Es4_cluster,E_abs_sq_inv_cluster=Esd4(p, sd_arr,win,cluster,var)

	ineff_c_sq=(2*Es4_cluster+max(((cluster/win),1))*((Es4_cluster)-1))/N
	ineff_sq=(3*Es4-1)/N
	ineff_r=(ineff_c_sq/ineff_sq)**0.5

	ineff_c_abs=(max(((cluster/win),1))*(E_abs_sq_inv_cluster-1)+E_abs_sq_inv_cluster*(np.pi*0.5-1))/N
	ineff_abs=((E_abs_sq_inv-1)+E_abs_sq_inv*(np.pi*0.5-1))/N
	ineff_abs_r=(ineff_c_abs/ineff_abs)**0.5


	return (ineff_abs_r,ineff_r)

def Esd4(p, sd_arr, win,cluster,v):
	eff_win=max((win/cluster,1))
	eff_win=int(np.round(eff_win,0))
	var,P=prob_space(p, sd_arr, eff_win)
	var=var/(eff_win)
	r=np.sum(P*var**2)/v**2

	E_abs_sq_inv=v/np.sum(P*var**0.5)**2

	return r, E_abs_sq_inv


def EsN(p, sd_arr, win,cluster,v):
	eff_win=max((win/cluster,1))
	eff_win=int(np.round(eff_win,0))
	var,P=prob_space(p, sd_arr, eff_win)
	Es1=np.sum(P*var**0.5)
	Es2=np.sum(P*var)


	return Es1, Es2


def prob_space(p,sd_arr,win):
	"""returns the set of outcomes (variances) var and its probabilities P"""
	sd_arrsq=sd_arr**2
	if type(sd_arr)==float or type(sd_arr)==np.float64 or win>450:
		var=np.sum(p*sd_arrsq)*win
		return var, 1
	K=len(sd_arr)
	a=np.rollaxis(np.indices([win+1]*(K-1)),0,K).reshape((win+1)**(K-1),K-1)
	a=a[np.sum(a,1)<=win]
	b=win-np.sum(a,1).reshape((len(a),1))
	Q=np.concatenate((a,b),1)
	P=multinominaldist(Q,p)
	var=np.sum(Q*sd_arrsq,1)
	return var,P

def E_abs_calc(var,P,win):
	E_sd=np.sum(P*var**0.5)/win**0.5

	return E_sd



def multinominaldist(q,p):
	a=special.gammaln(np.sum(q[0])+1)
	b=np.sum(special.gammaln(q+1),1)
	c=np.prod(p**q,1)

	return np.exp(a-b)*c



def random_mixed_normal(mu,N,sd_arr,p,cluster):
	"""sd_arr should be an array of standard deviations such that np.sum(p*np.array(sd_arr))=1"""
	if cluster>1:
		N_clusters=int(N/cluster)
		r=np.random.uniform(size=N_clusters).reshape((N_clusters,1))
		r=r*np.ones((1,cluster))
	else:
		r=np.random.uniform(size=N)
	r=r.reshape((N,1))
	p1=np.cumsum(p).reshape((1,len(p)))
	p1[0,2]=1.1
	p0=np.append([[-0.1]],p1[0,:-1])
	n=np.nonzero((r>p0)*(r<=p1))
	sd_rm=sd_arr[n[1]]
	rnd=np.random.normal(mu,sd_rm,N)

	return rnd

def return_calc(p,d,N,windows,win,p0):	
	if d==0:
		p_obs=p
	else:
		p0=np.round(p0/d)*d
		rnded=np.round(p/d)
		p_obs=rnded*d
	lp_obs=np.log(p_obs)

	lp_obs=lp_obs.reshape((windows,win)).T
	ret_period=lp_obs[-1]-np.roll(lp_obs[-1],1)
	ret_period[0]=lp_obs[-1,0]-np.log(p0)

	return lp_obs,ret_period




def sim_returns(mixed_norm,mu,sd,N,d,win,cluster,p,psd,sd_arr):
	windows=int(N/win)
	N=windows*win	

	if mixed_norm:
		rnd=random_mixed_normal(mu,N,sd_arr,p,cluster)
	else:
		m=0
		rnd=np.random.normal(mu,sd,N)

	#if rho>0: not in use, clustering used instead
	#	rnd=rnd-rho*np.roll(rnd,1)

	p0=1+np.random.uniform(-d, d)#ensures it does not allways start far from a changing value
	p_cont=np.exp(np.cumsum(rnd))
	p_cont=p_cont*p0

	lp_obs,ret_period=return_calc(p_cont,d,N,windows,win,p0)


	return lp_obs,ret_period,sd_arr,p,windows



def simulation(mu,sd,N,d,win,adj,mixed_norm,cluster,p,sd_arr,E_abs,E_abs2,pmsq,psd, avg_tz_bias,tz_bias):

	lp_obs,ret_period,sd_arr,p,windows=sim_returns(mixed_norm,mu,sd,N,d, win,cluster,p,psd,sd_arr)


	rng=rng_func(lp_obs,win,adj,mixed_norm,psd,cluster,p,sd_arr,E_abs2,sd)

	#msq_adj,msq_unadj,avg_abs,msq=msq_func(win,windows,ret_period,adj,pmsq,sd_arr,d,p,psd,E_abs, avg_tz_bias,tz_bias,sd)

	#return np.array([rng,msq_adj,msq_unadj,avg_abs,msq])

	r=msq_func(win,windows,ret_period,adj,pmsq,sd_arr,d,p,psd,E_abs, avg_tz_bias,tz_bias,sd)

	return [rng]+list(r)






def msq_func(win,windows,ret,adj,pmsq,sd_arr,d,p,psd,E_abs, avg_tz_bias,tz_bias,sd):
	if windows==1:
		return np.nan,np.nan
	freq=len(np.nonzero(ret!=0)[0])
	return calccovars(d,psd,ret,win,windows,1500)
	ret=ret-np.sum(ret)/(windows)
	ln_bias=(2*np.exp(special.psi((windows-1)/2))/(windows-1))**0.5

	msq=(np.sum(ret**2)/(win*(windows-1)))

	avg_abs=(np.sum(np.abs(ret))/win**0.5)/(((windows-1)*(windows))**0.5)
	
	dofadj=dof_adj(windows-1)
	abs_bias=(2/np.pi)**0.5/dofadj

	avg_abs=avg_abs/abs_bias
	if adj or False:#Set to true to correct for mixed normal
		mixedadj=(((windows-1)*E_abs**2+sd**2)/windows)**0.5
		avg_abs=avg_abs/mixedadj
	else:
		avg_abs=avg_abs/sd

	msq_adj=msq/(tz_bias*pmsq)**2
	msq_unadj=msq/pmsq**2
	msq_adj2=msq/(tz_bias*pmsq)**2

	return [msq_unadj,avg_abs]


def calccovars(d,sd,ret,win,windows,n):
	Pd=((2/np.pi)**0.5*sd/d)
	tz2bias=(d*(2/np.pi)**0.5*sd)
	tz4bias=(d**3*(2/np.pi)**0.5*sd)
	tz2bias=1
	if False:#
		var4=np.sum(ret**4)/(win*(windows-1)*tz2bias**2)		
		covars=[(np.sum((ret[i:]*ret[:-i])**2)/(win*(windows-i)*tz2bias**2)) for i in range(1,n)]
		i=10000
		cvN=np.sum((ret[i:]*ret[:-i])**2)/(win*(windows-i)*tz2bias**2)
		return [var4]+covars +[cvN] 
	if True:
		var4=np.sum(ret**2)/(win*(windows-1)*tz2bias)		
		covars=[(np.sum(((ret[i:]*ret[:-i]>0)*ret[i:]*ret[:-i])**2)/(win*(windows-i)*tz2bias**2)) for i in range(1,n)]
		return [var4]+covars
	covars=(np.sum((ret[n:]*ret[:-n])**2)/(win*(windows-n)*tz2bias**2))
	c=np.sum(ret!=0)/(Pd*len(ret))
	return [c,covars]	




def dof_adj(k,j=1):

	if k>250:
		a=1
	else:
		if k<1:
			k=1
		a=2**(j/2)*special.gamma((k+j)/2)/(special.gamma(k/2))	
		a=a/k**(j/2)
	return a


def rng_func(data,win, adj,mixed_norm,psd,cluster,p,sd_arr,E_abs,sd):
	if win==1:
		return np.nan	

	mx=np.max(data,0)
	mn=np.min(data,0)
	as_exp=(2/np.pi)**0.5
	rng=np.mean(mx-mn)/(as_exp*(win-1)**0.5)
	if not adj:
		return rng/sd

	a=np.exp(-(win-2)**0.5*0.267)
	ret=rng/(psd*(2-a))
	return ret



