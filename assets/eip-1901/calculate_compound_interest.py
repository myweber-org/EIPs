def calculate_compound_interest(principal, annual_rate, years, annual_contribution=0):
    """
    Calculate the future value of an investment with compound interest
    and optional annual contributions.
    
    Args:
        principal: Initial investment amount
        annual_rate: Annual interest rate as a decimal (e.g., 0.05 for 5%)
        years: Number of years to invest
        annual_contribution: Additional annual contribution (default: 0)
    
    Returns:
        Future value of the investment
    """
    future_value = principal
    
    for year in range(1, years + 1):
        # Add interest for the year
        future_value *= (1 + annual_rate)
        
        # Add annual contribution at the end of each year
        if year < years:  # Don't add contribution in the final year
            future_value += annual_contribution
    
    return round(future_value, 2)

def generate_investment_table(principal, annual_rate, years, annual_contribution=0):
    """
    Generate a year-by-year breakdown of investment growth.
    
    Args:
        principal: Initial investment amount
        annual_rate: Annual interest rate as a decimal
        years: Number of years to invest
        annual_contribution: Additional annual contribution
    
    Returns:
        List of tuples containing (year, value) pairs
    """
    table = []
    current_value = principal
    
    for year in range(years + 1):
        table.append((year, round(current_value, 2)))
        
        if year < years:
            # Calculate next year's value
            current_value *= (1 + annual_rate)
            current_value += annual_contribution
    
    return table

if __name__ == "__main__":
    # Example usage
    initial_investment = 10000
    interest_rate = 0.07
    investment_years = 20
    yearly_addition = 2000
    
    final_value = calculate_compound_interest(
        initial_investment, 
        interest_rate, 
        investment_years, 
        yearly_addition
    )
    
    print(f"Initial investment: ${initial_investment}")
    print(f"Annual interest rate: {interest_rate * 100}%")
    print(f"Investment period: {investment_years} years")
    print(f"Annual contribution: ${yearly_addition}")
    print(f"Final value: ${final_value:,}")
    
    # Generate and display table
    print("\nYear-by-year growth:")
    growth_table = generate_investment_table(
        initial_investment,
        interest_rate,
        investment_years,
        yearly_addition
    )
    
    for year, value in growth_table:
        if year % 5 == 0 or year == investment_years:
            print(f"  Year {year}: ${value:,}")