


class Functions:







	def DemagFactor(a,b,c):
	    '''
	        c must be parallel to the applied magnetic field.
	        
	        Finds the demagnetizing factor for a rectangular prism onto which a
	        magnetic field is applied. The field is assumed to be applied parallel
	        to the z-axis, and therefore perpendicularly to the x- and y-axes.
	        
	        The values of the input variables are taken to be half the length
	        of the rectangular prism, with one length for each spatial dimension.
	        -a < x < a
	        -b < y < b
	        -c < z < c
	        
	        Thus, each a, b, and c are half the sample length in each direction.

	        TODO: Make more readable.
	    '''
	    return (1/np.pi)*(((b**2 - c**2)*np.log((np.sqrt(a**2 + b**2 + c**2) - a)/(np.sqrt(a**2 + b**2 + c**2) + a)))/
      (2*b*c) + ((a**2 - c**2)*np.log((np.sqrt(a**2 + b**2 + c**2) - b)/(np.sqrt(a**2 + b**2 + c**2) + b)))/(2*a*c) + 
     (b*np.log((np.sqrt(a**2 + b**2) + a)/(np.sqrt(a**2 + b**2) - a)))/(2*c) + 
     (a*np.log((np.sqrt(a**2 + b**2) + b)/(np.sqrt(a**2 + b**2) - b)))/(2*c) + 
     (c*np.log((np.sqrt(b**2 + c**2) - b)/(np.sqrt(b**2 + c**2) + b)))/(2*a) + 
     (c*np.log((np.sqrt(a**2 + c**2) - a)/(np.sqrt(a**2 + c**2) + a)))/(2*b) + 2*np.arctan((a*b)/(c*np.sqrt(a**2 + b**2 + c**2))) + 
     (a**3 + b**3 - 2*c**3)/(3*a*b*c) + ((a**2 + b**2 - 2*c**2)*np.sqrt(a**2 + b**2 + c**2))/(3*a*b*c) + 
     (c*(np.sqrt(a**2 + c**2) + np.sqrt(b**2 + c**2)))/(a*b) - ((a**2 + b**2)**(3/2) + (b**2 + c**2)**(3/2) + 
       (c**2 + a**2)**(3/2))/(3*a*b*c))

	def LandeGFactor(J):
		'''
			https://en.wikipedia.org/wiki/Land%C3%A9_g-factor
		'''
	    L = J - S
	    top = S*(S+1)-L*(L+1)
	    bottom = 2*J*(J+1)

    	# The 1 on wiki comes from Kittel, Blundell has 3/2 but that is equivalent based on his second term.
	    g_j_Yb = 3/2 + top / bottom    
	    return g_j_Yb

	def BrillouinFunction_Yb_L(x, J):
			'''
				TODO: Check this against Blundell.
			'''
	        # J = L + S
	        # L = J - S


	        J1 = (2*J+1)/(2*J)
	        J2 = 1/(2*J)
	        
	        coth1 = 1/np.tanh(J1*x)
	        coth2 = 1/np.tanh(J2*x)

	        Bj= J1*coth1 - J2*coth2

	        return Bj

	        
class Constants:


	def muBohr(self):

		return
