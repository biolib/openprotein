import * as React from 'react';
import './App.css';
import Metrics from "./Metrics";
import Visualizer from "./Visualizer";

interface IAppProb {
    pdbData: {true: string, pred: string};
}

class App extends React.Component<IAppProb, IAppProb> {
  public state = { pdbData: this.props.pdbData };

  constructor(props: any) {
    super(props);
    this.setState({pdbData: {pred: "", true: ""}})
    this.setPdbData = this.setPdbData.bind(this);
  }

  public setPdbData(data: {true: string, pred: string}) {
    if (this.state.pdbData.true !== data.true || this.state.pdbData.pred !== data.pred) {
        this.setState({pdbData: data})
    }
  }

  public render() {
      const mainPanelStyle = {
          background: "white",
          display: "inline-grid" ,
          gridColumn: "1 / 3",
          gridRow: "2 / 30",

      };

      const leftPanelStyle = {
          background: "white",
          display: "inline-grid" ,
          gridColumn: "3",
          gridRow: "2 / 30",
      };

      const headerStyle = {
          background: "#222",
          display: "inline-grid" ,
          gridColumn: "1 / 4",
          gridRow: "1",
      };

      const footerStyle = {
          background: "#222",
          display: "inline-grid" ,
          gridColumn: "1 / 4",
          gridRow: "31 / 32",
      };

      return (
      <div className="App">
          <div style={{display: "grid" , height: "100vh",}}>
              <div style={headerStyle}>
                  <div style={{gridColumn: 1, color: "white", textAlign: "left", marginLeft:"20px"}}>
                      <h1 className="App-title">OpenProtein</h1>
                  </div>
                  <div style={{gridColumn: 2, color: "white", textAlign: "right"}}>
                      <div style={{margin: "20px", color: "white", fontSize:"10pt"}}>
                          <a style={{color: "white", textDecoration:"none"}} href="https://github.com/openprotein">View OpenProtein on Github</a>
                      </div>
                  </div>
              </div>
              <div style={mainPanelStyle}><Visualizer pdbData={this.state.pdbData} /></div>
              <div style={leftPanelStyle}><Metrics setPdbData={this.setPdbData} /></div>
              <div style={footerStyle}>
                  <div style={{gridColumn: 1, color: "white", textAlign: "left"}}>
                      <div style={{margin: "20px", display:"flex"}}>
                          <div style={{background:"blue", height:"12px", width:"20px", marginRight:"10px", marginLeft:"5px", border: "2px solid white"}} />
                          Prediction
                          <div style={{background:"green", height:"12px", width:"20px", marginRight:"10px", marginLeft:"15px", border: "2px solid white"}} />
                          Actual
                      </div>
                  </div>
                  <div style={{gridColumn: 2, color: "white", textAlign: "right"}}>
                      <div style={{margin: "20px", color: "lightgrey", fontSize:"10pt"}}>
                          © 2019 OpenProtein, all rights reserved
                      </div>
                  </div>
              </div>
          </div>
      </div>
    );
  }

  public componentDidMount(){
    document.title = "OpenProtein"
  }

}

export default App;
