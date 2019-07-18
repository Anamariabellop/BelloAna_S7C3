#Ejercicio 1 Terminar lo que hizo en clase + dos preguntas adicionales (en mayusculas en el texto)

import numpy as np
import matplotlib.pylab as plt


# 1) lea los datos de resorte.dat y almacenelos.
# 

datos=np.genfromtxt("resorte.dat")
t=datos[:,0]
yobs=datos[:,1]
#print(len(t),len(x))
# Los datos corresponden a las posiciones en x de un oscilador (masa resorte) en funcion del tiempo. La ecuacion de movimiento esta dada por  
# xt=a*np.exp(-gamma*t)*np.cos(omega*t)
# Donde a, gamma, y omega son parametros.



# 2) Implemente un algoritmo que le permita, por medio de estimacion bayesiana de parametros, encontrar los parametros correspondientes a los datos d. Para esto debe:
# 2a.) definir una funcion que reciba los parametros que se busca estimar y los datos de tiempo y retorne el modelo  
def model(a,g,o):
	return a*np.exp(-g*t)*np.cos(o*t)

# 2b.) Definir una funcion que retorne la funcion de verosimilitud

def verosimilitud(yobs,ymod):
	chi= np.sum((yobs-ymod)**2)
	return np.exp(-0.5*chi)

# 2c.) Caminata

#condiciones iniciales
aini=7.5
gammaini=0.6
omegaini=18.2

#numero de pasos
iteraciones=100000

def Bayes(aviejo, gviejo, omegaviejo,iterac,sigma):
	a=[]
	gamma=[]
	omega=[]
	L=[]

	a.append(aviejo)
	gamma.append(gviejo)
	omega.append(omegaviejo)

	ymod= model(aviejo,gviejo,omegaviejo)
	inicial= verosimilitud(yobs,ymod)

	L.append(inicial)

	for i in range(iterac):

		anuevo= np.random.normal(a[i],sigma)
		gnuevo= np.random.normal(gamma[i],sigma)
		omeganuevo= np.random.normal(omega[i],sigma)

		ymod= model(anuevo,gnuevo,omeganuevo)
		Lnuevo= verosimilitud(yobs,ymod)

		alpha= Lnuevo/L[i]

		if(alpha > 1):
			a.append(anuevo)
			gamma.append(gnuevo)
			omega.append(omeganuevo)
			L.append(Lnuevo)
		else:
			beta= np.random.uniform(0,1)

			if( beta < alpha):
				a.append(anuevo)
				gamma.append(gnuevo)
				omega.append(omeganuevo)
				L.append(Lnuevo)

			else:
				a.append(a[i])
				gamma.append(gamma[i])
				omega.append(omega[i])
				L.append(L[i])

	return a,gamma,omega,L

#print(Bayes(aini,gammaini,omegaini,iteraciones,0.1))

# 2d.) Seleccione los mejores parametros E IMPRIMA UN MENSAJE QUE DIGA: "LOS MEJORES PARAMETROS SON a=... gamma=... Y omgega=..."

paraseleccion= Bayes(aini,gammaini,omegaini,iteraciones,0.01)[3]

#saco el maximo de L

maximoLh= np.max(paraseleccion)
indice=np.argmax(paraseleccion)
#busco en indice de ese L para saber a, g, o

paraseleccion.index(maximoLh)

al=Bayes(aini,gammaini,omegaini,iteraciones,0.01)[0]
ga=Bayes(aini,gammaini,omegaini,iteraciones,0.01)[1]
om=Bayes(aini,gammaini,omegaini,iteraciones,0.01)[2]

abest=al[indice]
gbest=ga[indice]
obest=om[indice]

# 2f.) Grafique sus datos originales y su modelo con los mejores parametros. Guarde su grafica sin mostrarla en Resorte.pdf

plt.figure()
#plt.scatter(t,yobs, label='originales')
plt.plot(t,yobs, color='orchid', label='Original')
plt.plot(t,model(abest,gbest,obest), color='teal', label='Estimado')
plt.legend()
plt.grid()
plt.xlabel("t(s)")
plt.ylabel("Amplitud")
plt.title("Resorte")
plt.savefig("Resorte.pdf")

# 3) SABIENDO QUE omega=np.sqrt(k/m), IMPRIMA UN MENSAJE DONDE EXPLIQUE SI PUEDE O NO DETERMINAR k Y m DE MANERA INDIVIDUAL USANDO EL METODO ANTERIOR. JUSTIFIQUE BIEN SU RESPUESTA (PUEDE ADEMAS HACER PRUEBAS CON EL CODIGO PARA RESPONDER ESTA PREGUNTA).
print("  ")
print(" Si omega es la raiz de k/m se podria determinar individualmente volviendo a realizar el metodo de estimacion bayesiana a traves de los resultados obtenidos anteriormente pero en este caso, se debería estimar las variables k y m comparadas con las k y m que se obtengan con la funcion inicial(despeje de ecuacion), siempre y cuando se obtengan condiciones iniciales o valores iniciales de k y m.")
print("  ")
