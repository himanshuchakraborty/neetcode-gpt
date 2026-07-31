class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # f = x**2
        x = init
        for i in range(iterations):
        # Derivative:         f'(x) = 2x
            df_dx = 2*x
                
        # Update rule:        x = x - learning_rate * f'(x)
            x -= learning_rate*df_dx

        # Round final answer to 5 decimal places
        return round(x,5)
print(Solution.get_minimizer(0, 5, 0.001, 1))
        
