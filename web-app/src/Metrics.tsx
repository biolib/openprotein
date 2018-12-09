
import * as Chart from 'chart.js'
import * as $ from 'jquery'
// @ts-ignore
import * as NGL from 'ngl'
import * as React from 'react';
import './App.css';

interface IMetricsProbs {
    setPdbData: any;
}

class Metrics extends React.Component<IMetricsProbs, any> {

    // @ts-ignore
    private setPdbData : any;

    constructor(props: any){
        super(props);
        this.setPdbData = this.props.setPdbData;
    }

    // @ts-ignore
    public componentDidMount() {

        // @ts-ignore
        const ctx = document.getElementById("myChart").getContext('2d');
        // @ts-ignore
        const ctx2 = document.getElementById("myChart2").getContext('2d');

        const lineConfig =  {
            data: {
                datasets: [{
                    backgroundColor: 'rgb(255, 99, 132)',
                    borderColor: 'rgb(255, 99, 132)',
                    data: [],
                    fill: false,
                    label: 'Train Loss',
                    yAxisID: 'y-axis-1'
                }, {
                    backgroundColor: 'rgb(54, 162, 235)',
                    borderColor: 'rgb(54, 162, 235)',
                    data: [],
                    fill: false,
                    label: 'Validation dRMSD avg',
                    yAxisID: 'y-axis-2'
                }, {
                    backgroundColor: 'rgb(75, 192, 192)',
                    borderColor: 'rgb(75, 192, 192)',
                    data: [],
                    fill: false,
                    label: 'Validation RMSD avg',
                    yAxisID: 'y-axis-3'
                }],
                labels: [],
            },
            options: {
                hover: {
                    intersect: true,
                    mode: 'nearest',
                },
                responsive: true,
                scales: {
                    xAxes: [{
                        display: true,
                        scaleLabel: {
                            display: true,
                            labelString: 'Minibatch updates'
                        }
                    }],
                    yAxes: [{
                        display: true,
                        id: 'y-axis-1',
                        position: 'left',
                        scaleLabel: {
                            display: true,
                            labelString: 'Loss1'
                        },
                        type: 'linear', // only linear but allow scale type registration. This allows extensions to exist solely for log scale for instance
                    }, {
                        display: true,
                        gridLines: {
                            drawOnChartArea: false, // only want the grid lines for one axis to show up
                        },
                        id: 'y-axis-2',
                        position: 'right',
                        scaleLabel: {
                            display: true,
                            labelString: 'dRMSD'
                        },
                        type: 'linear', // only linear but allow scale type registration. This allows extensions to exist solely for log scale for instance
                    }, {
                        display: true,
                        gridLines: {
                            drawOnChartArea: false, // only want the grid lines for one axis to show up
                        },
                        id: 'y-axis-3',
                        position: 'right',
                        scaleLabel: {
                            display: true,
                            labelString: 'RMSD'
                        },
                        type: 'linear', // only linear but allow scale type registration. This allows extensions to exist solely for log scale for instance
                    }]
                },
                title: {
                    display: true,
                    text: 'Training Progress'
                },
                tooltips: {
                    intersect: false,
                    mode: 'index',
                },
            },
            type: 'line',
        };

        // @ts-ignore
        const myChart = new Chart(ctx, lineConfig);


        const scatterConfig = {
            data: {
                datasets: [{
                    backgroundColor: 'rgb(255, 99, 132)',
                    borderColor: 'rgb(255, 99, 132)',
                    data: [],
                    label: 'Actual',
                },{
                    backgroundColor: 'rgb(54, 162, 235)',
                    borderColor: 'rgb(54, 162, 235)',
                    data: [],
                    label: 'Predicted',
                }]
            },
            options: {
                scales: {
                    xAxes: [{
                        display: true,
                        scaleLabel: {
                            display: true,
                            labelString: 'Phi'
                        }
                    }],
                    yAxes: [{
                        display: true,
                        id: 'y-axis-1',
                        position: 'left',
                        scaleLabel: {
                            display: true,
                            labelString: 'Psi'
                        },
                        type: 'linear', // only linear but allow scale type registration. This allows extensions to exist solely for log scale for instance

                    }]
                },
                title: {
                    display: true,
                    text: 'Ramachandran Plot'
                },
            }
        };
        // @ts-ignore
        const myChart2 = Chart.Scatter(ctx2, scatterConfig);

        let connectionMade = false;


        function update_data(app: any) {
            $.getJSON( "http://localhost:5000/graph", ( data ) => {
                connectionMade = true;
                if (data == null) {
                    // @ts-ignore
                    // console.log("Data is null, returning...");
                    return;
                }

                app.setPdbData({true: data.pdb_data_true, pred: data.pdb_data_pred});

                scatterConfig.data.datasets[0].data = data.phi_actual.map( (h: any, i: any) => {
                    return {
                        x: h,
                        y: data.psi_actual[i],
                    };
                });
                scatterConfig.data.datasets[1].data = data.phi_predicted.map( (h: any, i: any) => {
                    return {
                        x: h,
                        y: data.psi_predicted[i],
                    };
                });

                lineConfig.data.labels = data.sample_num
                lineConfig.data.datasets[0].data = data.sample_num.map( (h: any, i: any) => {
                    return {
                        x: h,
                        y: data.train_loss_values[i],
                    };
                });
                lineConfig.data.datasets[1].data = data.sample_num.map( (h: any, i: any) => {
                    return {
                        x: h,
                        y: data.drmsd_avg[i],
                    };
                });
                lineConfig.data.datasets[2].data = data.sample_num.map( (h: any, i: any) => {
                    return {
                        x: h,
                        y: data.rmsd_avg[i],
                    };
                });

                myChart.update();
                myChart2.update();
            } );
        }

        const updateDataInterval = window.setInterval(update_data, 1000, this);

        $.ajaxSetup({
            "error":() => {
                if (connectionMade === true) {
                    clearInterval(updateDataInterval)
                    // console.log("Connection to server lost, NOT retrying")
                }
            }
        });

    }

  public render() {
      return (
          <div className="Metrics">
              <div className="chart-container" style={{width: "40vw", height: "100%"}}>
                  <canvas id="myChart"  />
                  <canvas id="myChart2"  />
              </div>
          </div>
    );
  }
}

export default Metrics;
