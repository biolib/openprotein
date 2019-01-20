
// @ts-ignore
import * as NGL from 'ngl'
import * as React from 'react';
import './App.css';

interface IVisualizeProb {
    pdbData: {true: string, pred: string};
}

class Visualizer extends React.Component<IVisualizeProb,any> {

    private stage = null;

    public componentDidUpdate() {
        if (this.stage == null) {
            this.stage = new NGL.Stage("viewport", { backgroundColor: "white" })
            // @ts-ignore
            this.stage.toggleSpin()
        }
            const stringBlobPred = new Blob( [ this.props.pdbData.pred ], { type: 'text/plain'} );
            const stringBlobTrue = new Blob( [ this.props.pdbData.true ], { type: 'text/plain'} );

        // @ts-ignore
        this.stage.removeAllComponents()

        const schemeId = NGL.ColormakerRegistry.addSelectionScheme([
            ["blue", "*"]
        ], "Transmembrane 3dqb");
        const schemeIdGreen = NGL.ColormakerRegistry.addSelectionScheme([
            ["green", "*"]
        ], "Transmembrane 3dqb");
        // @ts-ignore
        this.stage.loadFile( stringBlobPred, { ext: "pdb", defaultRepresentation: false } ).then( ( structureComponent ) => {
            structureComponent.addRepresentation("cartoon", {color: schemeId });
            structureComponent.autoView();
        } );;
        // @ts-ignore
        this.stage.loadFile( stringBlobTrue, { ext: "pdb", defaultRepresentation: false } ).then( ( structureComponent ) => {
            structureComponent.addRepresentation("cartoon", {color: schemeIdGreen });
            structureComponent.autoView();
        } );

    }

  public render() {
      return (
          <div className="Visualizer" style={{width: "50vw", height: "100%"}}>
              <div id="viewport" style={{width: "100%", height: "100%"}} />
          </div>
    );
  }
}

export default Visualizer;
