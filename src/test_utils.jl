module TestUtils

using CSV
using DataFrames
using HTTP
using OrderedCollections

financial_url = "https://raw.githubusercontent.com/scikit-learn/examples-data/master/financial-data/"

financial_dict = OrderedDict(
    "TOT" => "Total",
    "XOM" => "Exxon",
    "CVX" => "Chevron",
    "COP" => "ConocoPhillips",
    "VLO" => "Valero Energy",
    "MSFT" => "Microsoft",
    "IBM" => "IBM",
    "TWX" => "Time Warner",
    "CMCSA" => "Comcast",
    "CVC" => "Cablevision",
    "YHOO" => "Yahoo",
    "DELL" => "Dell",
    "HPQ" => "HP",
    "AMZN" => "Amazon",
    "TM" => "Toyota",
    "CAJ" => "Canon",
    "SNE" => "Sony",
    "F" => "Ford",
    "HMC" => "Honda",
    "NAV" => "Navistar",
    "NOC" => "Northrop Grumman",
    "BA" => "Boeing",
    "KO" => "Coca Cola",
    "MMM" => "3M",
    "MCD" => "McDonald\"s",
    "PEP" => "Pepsi",
    "K" => "Kellogg",
    "UN" => "Unilever",
    "MAR" => "Marriott",
    "PG" => "Procter Gamble",
    "CL" => "Colgate-Palmolive",
    "GE" => "General Electrics",
    "WFC" => "Wells Fargo",
    "JPM" => "JPMorgan Chase",
    "AIG" => "AIG",
    "AXP" => "American express",
    "BAC" => "Bank of America",
    "GS" => "Goldman Sachs",
    "AAPL" => "Apple",
    "SAP" => "SAP",
    "CSCO" => "Cisco",
    "TXN" => "Texas Instruments",
    "XRX" => "Xerox",
    "WMT" => "Wal-Mart",
    "HD" => "Home Depot",                                                                    
    "GSK" => "GlaxoSmithKline",                                                              
    "PFE" => "Pfizer",                                                                       
    "SNY" => "Sanofi-Aventis",                                                               
    "NVS" => "Novartis",                                                                     
    "KMB" => "Kimberly-Clark",                                                               
    "R" => "Ryder",                                                                          
    "GD" => "General Dynamics",                                                              
    "RTN" => "Raytheon",                                                                     
    "CVS" => "CVS",                                                                          
    "CAT" => "Caterpillar",                                                                  
    "DD" => "DuPont de Nemours"                                                              
)                                                                                            

function get_financial_data(; D::Int=56, N::Int=1258)
    financial_data = DataFrame()
    for key in collect(keys(financial_dict))[1:D]
        financial_data[!, key] = DataFrame(CSV.File(HTTP.get(financial_url*key*".csv").body))[!, :close]
    end
    return financial_data[1:N, :]
end

end
