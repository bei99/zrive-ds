import openmeteo_requests
import pandas as pd
import matplotlib.pyplot as plt
import retry


def get_data_meteo_api(
        cities_with_coords: dict, weather_variables:list, start_date: str, end_date:str) -> dict:

    url = "https://archive-api.open-meteo.com/v1/archive?"
    openmeteo = openmeteo_requests.Client()
    daily_data = {
    "date": pd.date_range(
        start=start_date,
        end=end_date,
        freq="1D"
    )}

    df = pd.DataFrame(daily_data)


    for city, coords in cities_with_coords.items():
        params = {
        "latitude": round(coords["latitude"],2),
        "longitude": round(coords["longitude"],2),
        "start_date": start_date,
        "end_date": end_date,
        "daily": weather_variables}

        responses = openmeteo.weather_api(url,params)
        response = responses[0]
        daily = response.Daily()


        for i in range(len(weather_variables)):
            data = pd.DataFrame({f"{city}_{weather_variables[i]}": daily.Variables(i).ValuesAsNumpy()})
            df = pd.concat([df, data], axis=1)


   
    return df

def process_meteo_time_series(df):

    df['year_month'] = df['date'].dt.strftime('%Y-%m')
    df = df.drop(columns=['date'])

    temperature_columns = df.filter(regex='temperature').columns
    precipitation_columns = df.filter(regex='precipitation').columns
    wind_columns = df.filter(regex='wind').columns

    monthly_agg = df.groupby(['year_month']).agg({
        **{col:'mean' for col in temperature_columns},
        **{col:'sum' for col in precipitation_columns},
        **{col:'max' for col in wind_columns}}).reset_index()
    
    monthly_agg.iloc[:, 1:] = monthly_agg.iloc[:, 1:].round(2)

    return monthly_agg

def plot_meteo_temperature_time_series(df):
    variables = {"temperature":{"metric":"Average","measure":"°C"},
                    "precipitation":{"metric":"Total","measure":"mm"},
                    "wind":{"metric":"Max","measure":"km/h"}}
        
    for var,details in variables.items():
        metric = details["metric"]
        measure = details["measure"]

        variable_columns = df.filter(regex=var).columns.tolist()

        temp_df = df[['year_month'] + variable_columns]

        temp_df = temp_df.rename(columns={
            temp_df.columns[0]:'year_month',
            temp_df.columns[1]:'Madrid',
            temp_df.columns[2]:'London',
            temp_df.columns[3]:'Rio'})     
        temp_df.set_index('year_month', inplace=True)

      

        plt.figure(figsize=(25, 6))
        temp_plot = temp_df.plot(
            kind='line',
            title=f"Monthly {metric} {var} in {measure} (2010-2020)",
            marker='o',
            markersize=2,
            color=['red', 'blue', 'green'])

        plt.xticks(
            ticks=range(0, len(temp_df.index),6),  
            labels=temp_df.index[::6],             
            rotation=90
        )
        
        plt.grid(visible=True, linestyle='--', alpha=0.6)

        plt.xlabel('Month')
        plt.ylabel(f"{var} in {measure}")
        plt.legend(title='City')

        filename = f"zrive_ds/src/module_1/results/{var}_plot.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')  
        print(f"Plot saved as {filename}")

def main():
    cities = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896}}

    variables = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]

    start_date = "2010-01-01"

    end_date = "2020-12-31"

    raw_data = get_data_meteo_api(cities, variables, start_date, end_date)

    processed_data = process_meteo_time_series(raw_data)

    plot_meteo_temperature_time_series(processed_data)



if __name__ == "__main__":
    main()
