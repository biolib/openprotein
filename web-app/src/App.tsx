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

      return (
      <div className="App">
          <div style={{display: "grid" , height: "100vh",}}>
              <div style={headerStyle}> <h1 className="App-title">Welcome to OpenProtein</h1></div>
              <div style={mainPanelStyle}><Visualizer pdbData={this.state.pdbData} /></div>
              <div style={leftPanelStyle}><Metrics setPdbData={this.setPdbData} /></div>
          </div>
      </div>
    );
  }
}

export default App;
