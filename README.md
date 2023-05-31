# Summary of changes to Main branch:

April 5, 2023 - Made sigma (risk averseness) heterogeneous among agents between an arbitrary range from 0.9 to 1.9; the original value was 1.5.

April 10, 2023 - Generalized functions income_generation, isocline, and income_updation and calculations for self.front(s) and self.slope to work with any number of tech options housed in a dictionary, TechTable.

May 31, 2023 - Changed the consumption calculation method to optimization with scipy. 
             - Added agent attribute theta, which is their personal perception of future shocks. This value is updated at every time step based on their sensitivity to observed theta.
